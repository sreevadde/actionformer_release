#!/usr/bin/env python3
"""
ActionFormer Multi-GPU Training with DistributedDataParallel (DDP)

Features:
    - DDP for efficient multi-GPU training
    - AMP (Automatic Mixed Precision) for faster training
    - Gradient accumulation for larger effective batch sizes
    - Distributed validation/evaluation

Usage:
    # Basic DDP training on 4 GPUs
    torchrun --nproc_per_node=4 train_ddp.py configs/your_config.yaml

    # With AMP (mixed precision) - ~2x faster training
    torchrun --nproc_per_node=4 train_ddp.py configs/your_config.yaml --amp

    # With gradient accumulation (effective batch = batch_size * accum_steps * num_gpus)
    torchrun --nproc_per_node=4 train_ddp.py configs/your_config.yaml --accum-steps 4

    # Full example with all features
    torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml \\
        --amp --accum-steps 2 --output my_experiment

Multi-node training:
    # On node 0 (master)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
        --master_addr=<master_ip> --master_port=29500 \\
        train_ddp.py configs/your_config.yaml --amp

    # On node 1
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
        --master_addr=<master_ip> --master_port=29500 \\
        train_ddp.py configs/your_config.yaml --amp
"""

import argparse
import os
import time
import datetime
from pprint import pprint
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, AverageMeter)


################################################################################
# DDP Utilities
################################################################################

def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return dist.get_rank() == 0


def get_world_size():
    """Get the total number of processes."""
    return dist.get_world_size()


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes (average)."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


################################################################################
# Training Functions
################################################################################

def train_one_epoch_ddp(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    local_rank,
    model_ema=None,
    clip_grad_l2norm=-1,
    tb_writer=None,
    print_freq=20,
    use_amp=False,
    scaler=None,
    accum_steps=1
):
    """Training the model for one epoch with DDP.

    Supports:
        - Automatic Mixed Precision (AMP) for faster training
        - Gradient accumulation for larger effective batch sizes
        - Distributed logging and metrics

    Args:
        train_loader: DataLoader with DistributedSampler
        model: DDP-wrapped model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        curr_epoch: Current epoch number
        local_rank: Local GPU rank for this process
        model_ema: Optional ModelEMA instance (only used on rank 0)
        clip_grad_l2norm: Gradient clipping norm (-1 to disable)
        tb_writer: TensorBoard writer (only used on rank 0)
        print_freq: Print frequency in iterations
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP (required if use_amp=True)
        accum_steps: Number of gradient accumulation steps
    """
    world_size = get_world_size()
    is_main = is_main_process()

    # Set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    num_iters = len(train_loader)

    # Switch to train mode
    model.train()

    if is_main:
        print(f"\n[Train]: Epoch {curr_epoch} started")
        if use_amp:
            print("[Train]: Using AMP (mixed precision)")
        if accum_steps > 1:
            print(f"[Train]: Gradient accumulation steps: {accum_steps}")

    start = time.time()

    for iter_idx, video_list in enumerate(train_loader):
        # Determine if this is an accumulation step
        is_accumulating = (iter_idx + 1) % accum_steps != 0

        # Context manager for gradient sync control
        # Skip gradient sync during accumulation steps for efficiency
        if is_accumulating:
            ctx = model.no_sync()
        else:
            ctx = nullcontext()

        with ctx:
            if use_amp:
                # AMP forward pass
                with torch.cuda.amp.autocast():
                    losses = model(video_list)
                    loss = losses['final_loss'] / accum_steps

                # AMP backward pass
                scaler.scale(loss).backward()
            else:
                # Standard forward/backward
                losses = model(video_list)
                loss = losses['final_loss'] / accum_steps
                loss.backward()

        # Step optimizer after accumulation
        if not is_accumulating or (iter_idx + 1) == num_iters:
            if use_amp:
                # Gradient clipping with AMP
                if clip_grad_l2norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        clip_grad_l2norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard gradient clipping
                if clip_grad_l2norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        clip_grad_l2norm
                    )
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if model_ema is not None:
                model_ema.update(model)

        # Logging (only on main process, at print_freq intervals)
        if is_main and (iter_idx != 0) and (iter_idx % print_freq) == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # Track losses
            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())

            # TensorBoard logging
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx

            if tb_writer is not None:
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars('train/all_losses', tag_dict, global_step)
                tb_writer.add_scalar('train/final_loss',
                                     losses_tracker['final_loss'].val, global_step)

            # Terminal output
            block1 = f'Epoch: [{curr_epoch:03d}][{iter_idx:05d}/{num_iters:05d}]'
            block2 = f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'
            block3 = f'Loss {losses_tracker["final_loss"].val:.2f} ({losses_tracker["final_loss"].avg:.2f})'

            loss_details = []
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    loss_details.append(f'{key} {value.val:.2f} ({value.avg:.2f})')

            print(f'\t{block1}\t{block2}\t{block3}')
            if loss_details:
                print(f'\t\t' + '\t'.join(loss_details))

    if is_main:
        lr = scheduler.get_last_lr()[0]
        print(f"[Train]: Epoch {curr_epoch} finished with lr={lr:.8f}\n")

    return


def valid_one_epoch_ddp(
    val_loader,
    model,
    curr_epoch,
    evaluator=None,
    tb_writer=None,
    print_freq=20
):
    """Validation for one epoch with DDP.

    Gathers results from all processes and evaluates on main process.

    Args:
        val_loader: Validation DataLoader (with DistributedSampler)
        model: DDP-wrapped model
        curr_epoch: Current epoch number
        evaluator: ANETdetection evaluator
        tb_writer: TensorBoard writer (only used on rank 0)
        print_freq: Print frequency

    Returns:
        mAP on main process, None on other processes
    """
    world_size = get_world_size()
    is_main = is_main_process()

    # Switch to eval mode
    model.eval()

    if is_main:
        print(f"\n[Eval]: Epoch {curr_epoch} started")

    all_results = []

    with torch.no_grad():
        for iter_idx, video_list in enumerate(val_loader):
            # Forward pass
            output = model(video_list)

            # Collect results
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    all_results.append({
                        'video-id': video_list[vid_idx]['video_id'],
                        'segments': output[vid_idx]['segments'].cpu(),
                        'scores': output[vid_idx]['scores'].cpu(),
                        'labels': output[vid_idx]['labels'].cpu()
                    })

            if is_main and (iter_idx + 1) % print_freq == 0:
                print(f'[Eval]: [{iter_idx + 1:05d}/{len(val_loader):05d}]')

    # Gather results from all processes
    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, all_results)

    # Evaluate on main process
    mAP = None
    if is_main and evaluator is not None:
        # Flatten results from all processes
        combined_results = {}
        for proc_results in gathered_results:
            for result in proc_results:
                vid_id = result['video-id']
                if vid_id not in combined_results:
                    combined_results[vid_id] = result
                # Note: Results for same video from different processes are identical
                # due to deterministic inference, so we just keep one

        # Format for evaluator
        results_list = list(combined_results.values())

        # Update evaluator and compute mAP
        for result in results_list:
            evaluator.update(
                result['video-id'],
                result['segments'].numpy(),
                result['scores'].numpy(),
                result['labels'].numpy()
            )

        mAP, _ = evaluator.evaluate()

        if tb_writer is not None:
            tb_writer.add_scalar('eval/mAP', mAP, curr_epoch)

        print(f"[Eval]: Epoch {curr_epoch} mAP = {mAP:.4f}")

    return mAP


################################################################################
# Helper context manager
################################################################################

class nullcontext:
    """A context manager that does nothing (for Python < 3.7 compatibility)."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


################################################################################
# Main
################################################################################

def trivial_batch_collator(batch):
    """Trivial collator that returns the batch as-is (list of dicts)."""
    return batch


def main(args):
    """Main function for DDP training."""

    # Setup DDP
    local_rank = setup_ddp()
    world_size = get_world_size()

    if is_main_process():
        print(f"=" * 60)
        print(f"ActionFormer DDP Training")
        print(f"=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"AMP: {'Enabled' if args.amp else 'Disabled'}")
        print(f"Gradient accumulation: {args.accum_steps} steps")
        print(f"Effective batch size: batch_size × {args.accum_steps} × {world_size}")
        print(f"=" * 60)

    # Load config
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if is_main_process():
        pprint(cfg)

    # Setup output folder
    if not os.path.exists(cfg['output_folder']):
        if is_main_process():
            os.makedirs(cfg['output_folder'], exist_ok=True)
    dist.barrier()

    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_ddp_' + str(ts))
    else:
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(args.output))

    if is_main_process():
        os.makedirs(ckpt_folder, exist_ok=True)
        tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))
    else:
        tb_writer = None
    dist.barrier()

    # Fix random seed (with rank offset for diversity)
    rng_generator = fix_random_seed(cfg['init_rand_seed'] + local_rank, include_cuda=True)

    # Scale learning rate by world size × accumulation steps (linear scaling rule)
    effective_batch_scale = world_size * args.accum_steps
    cfg['opt']["learning_rate"] *= effective_batch_scale
    if is_main_process():
        print(f"[Config]: Learning rate scaled by {effective_batch_scale}x to {cfg['opt']['learning_rate']}")

    # Workers per process
    cfg['loader']['num_workers'] = max(1, cfg['loader']['num_workers'] // world_size)

    # Create training dataset
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # Create distributed sampler for training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True
    )

    # Create training data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['loader']['batch_size'],
        shuffle=False,
        num_workers=cfg['loader']['num_workers'],
        collate_fn=trivial_batch_collator,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    # Create validation dataset and loader (if validation is enabled)
    val_loader = None
    evaluator = None
    if args.eval_freq > 0 and 'val_split' in cfg:
        val_dataset = make_dataset(
            cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg['loader']['batch_size'],
            shuffle=False,
            num_workers=cfg['loader']['num_workers'],
            collate_fn=trivial_batch_collator,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False
        )

        # Setup evaluator
        val_db_vars = val_dataset.get_attributes()
        evaluator = ANETdetection(
            val_dataset.data_list,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )

    # Create model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = model.to(local_rank)

    # Wrap with DDP (find_unused_parameters for models with optional branches)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    # Optimizer and scheduler
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # Model EMA (only on main process)
    model_ema = None
    if is_main_process():
        print("[Config]: Using model EMA")
        model_ema = ModelEma(model)

    # AMP setup
    scaler = None
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                print(f"[Resume]: Loading checkpoint '{args.resume}'")

            map_location = {'cuda:0': f'cuda:{local_rank}'}
            checkpoint = torch.load(args.resume, map_location=map_location)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            if model_ema is not None and 'state_dict_ema' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['state_dict_ema'])

            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])

            if is_main_process():
                print(f"[Resume]: Loaded checkpoint epoch {checkpoint['epoch']}")
            del checkpoint
        else:
            if is_main_process():
                print(f"[Resume]: No checkpoint found at '{args.resume}'")

    # Save config
    if is_main_process():
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(cfg, stream=fid)
            fid.write(f"\n\n# DDP Settings\n")
            fid.write(f"world_size: {world_size}\n")
            fid.write(f"amp: {args.amp}\n")
            fid.write(f"accum_steps: {args.accum_steps}\n")

    # Training loop
    if is_main_process():
        print(f"\n{'=' * 60}")
        print(f"Starting training...")
        print(f"{'=' * 60}")

    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    best_mAP = 0.0

    for epoch in range(args.start_epoch, max_epochs):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        # Train one epoch
        train_one_epoch_ddp(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            local_rank,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
            use_amp=args.amp,
            scaler=scaler,
            accum_steps=args.accum_steps
        )

        # Validation (if enabled)
        if val_loader is not None and (epoch + 1) % args.eval_freq == 0:
            mAP = valid_one_epoch_ddp(
                val_loader,
                model,
                epoch,
                evaluator=deepcopy(evaluator),  # Fresh evaluator each time
                tb_writer=tb_writer,
                print_freq=args.print_freq
            )

            if is_main_process() and mAP is not None:
                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                if is_best:
                    print(f"[Eval]: New best mAP: {best_mAP:.4f}")

        # Save checkpoint (only on main process)
        if is_main_process():
            if ((epoch + 1) == max_epochs) or \
               ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)):
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if model_ema is not None:
                    save_states['state_dict_ema'] = model_ema.module.state_dict()
                if scaler is not None:
                    save_states['scaler'] = scaler.state_dict()

                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name=f'epoch_{epoch + 1:03d}.pth.tar'
                )

        dist.barrier()

    # Cleanup
    if is_main_process():
        tb_writer.close()
        print(f"\n{'=' * 60}")
        print("Training complete!")
        if best_mAP > 0:
            print(f"Best mAP: {best_mAP:.4f}")
        print(f"{'=' * 60}")

    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DDP Training for ActionFormer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic 4-GPU training
    torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml

    # With AMP (faster training)
    torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml --amp

    # With gradient accumulation
    torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml --accum-steps 4

    # Full example
    torchrun --nproc_per_node=4 train_ddp.py configs/thumos_i3d.yaml \\
        --amp --accum-steps 2 --eval-freq 5 --output my_exp
        """)
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('-e', '--eval-freq', default=0, type=int,
                        help='validation frequency (default: 0 = disabled)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--amp', action='store_true',
                        help='use automatic mixed precision (AMP)')
    parser.add_argument('--accum-steps', default=1, type=int,
                        help='gradient accumulation steps (default: 1)')
    args = parser.parse_args()
    main(args)
