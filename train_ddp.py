#!/usr/bin/env python3
"""
ActionFormer Multi-GPU Training with DistributedDataParallel (DDP)

Usage:
    torchrun --nproc_per_node=4 train_ddp.py configs/your_config.yaml

This uses DDP which is more efficient than DataParallel for multi-GPU training.
DDP runs each GPU in its own process, avoiding Python GIL issues and providing
better scaling for models that receive non-tensor inputs (like list of dicts).
"""

import argparse
import os
import time
import datetime
from pprint import pprint

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
from libs.utils import (train_one_epoch_ddp, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


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


def main(args):
    """Main function for DDP training."""

    # Setup DDP
    local_rank = setup_ddp()
    world_size = dist.get_world_size()

    if is_main_process():
        print(f"Starting DDP training with {world_size} GPUs")

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
    dist.barrier()  # Wait for folder creation

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

    # Scale learning rate by world size (linear scaling rule)
    cfg['opt']["learning_rate"] *= world_size
    # Workers per process
    cfg['loader']['num_workers'] = max(1, cfg['loader']['num_workers'] // world_size)

    # Create dataset
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True
    )

    # Create data loader with distributed sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['loader']['batch_size'],
        shuffle=False,  # Sampler handles shuffling
        num_workers=cfg['loader']['num_workers'],
        collate_fn=trivial_batch_collator,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    # Create model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = model.to(local_rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer and scheduler
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # Model EMA (only on main process)
    if is_main_process():
        print("Using model EMA ...")
    model_ema = ModelEma(model) if is_main_process() else None

    # Resume from checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                print(f"Loading checkpoint '{args.resume}'")

            # Map to current GPU
            map_location = {'cuda:0': f'cuda:{local_rank}'}
            checkpoint = torch.load(args.resume, map_location=map_location)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            if model_ema is not None and 'state_dict_ema' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['state_dict_ema'])

            if is_main_process():
                print(f"Loaded checkpoint epoch {checkpoint['epoch']}")
            del checkpoint
        else:
            if is_main_process():
                print(f"No checkpoint found at '{args.resume}'")

    # Save config
    if is_main_process():
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(cfg, stream=fid)

    # Training loop
    if is_main_process():
        print(f"\nStart training model {cfg['model_name']} with DDP on {world_size} GPUs...")

    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

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
            print_freq=args.print_freq
        )

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
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name=f'epoch_{epoch + 1:03d}.pth.tar'
                )

        dist.barrier()  # Sync all processes

    # Cleanup
    if is_main_process():
        tb_writer.close()
        print("All done!")

    cleanup_ddp()


def trivial_batch_collator(batch):
    """
    Trivial collator that returns the batch as-is (list of dicts).
    This matches the original ActionFormer data loading.
    """
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DDP Training for ActionFormer')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
