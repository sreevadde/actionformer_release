#!/usr/bin/env python3
"""
ActionFormer Multi-GPU Training with DistributedDataParallel (DDP)

Usage:
    actionformer-train-ddp configs/thumos_i3d.yaml --amp --output my_exp

    # Or via torchrun:
    torchrun --nproc_per_node=4 -m actionformer.train_ddp configs/thumos_i3d.yaml --amp
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
from libs.utils import (
    valid_one_epoch, ANETdetection,
    save_checkpoint, make_optimizer, make_scheduler,
    fix_random_seed, ModelEma, AverageMeter
)


class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size()


def trivial_batch_collator(batch):
    return batch


def train_one_epoch_ddp(
    train_loader, model, optimizer, scheduler, curr_epoch, local_rank,
    model_ema=None, clip_grad_l2norm=-1, tb_writer=None, print_freq=20,
    use_amp=False, scaler=None, accum_steps=1
):
    world_size = get_world_size()
    is_main = is_main_process()
    batch_time = AverageMeter()
    losses_tracker = {}
    num_iters = len(train_loader)
    model.train()

    if is_main:
        print(f"\n[Train]: Epoch {curr_epoch} started")

    start = time.time()

    for iter_idx, video_list in enumerate(train_loader):
        is_accumulating = (iter_idx + 1) % accum_steps != 0
        ctx = model.no_sync() if is_accumulating else nullcontext()

        with ctx:
            if use_amp:
                with torch.cuda.amp.autocast():
                    losses = model(video_list)
                    loss = losses['final_loss'] / accum_steps
                scaler.scale(loss).backward()
            else:
                losses = model(video_list)
                loss = losses['final_loss'] / accum_steps
                loss.backward()

        if not is_accumulating or (iter_idx + 1) == num_iters:
            if use_amp:
                if clip_grad_l2norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad_l2norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if model_ema is not None:
                model_ema.update(model)

        if is_main and (iter_idx != 0) and (iter_idx % print_freq) == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())

            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx

            if tb_writer is not None:
                tb_writer.add_scalar('train/learning_rate', lr, global_step)
                tb_writer.add_scalar('train/final_loss', losses_tracker['final_loss'].val, global_step)

            block1 = f'Epoch: [{curr_epoch:03d}][{iter_idx:05d}/{num_iters:05d}]'
            block2 = f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'
            block3 = f'Loss {losses_tracker["final_loss"].val:.4f} ({losses_tracker["final_loss"].avg:.4f})'
            print(f'\t{block1}\t{block2}\t{block3}')


def run(args):
    local_rank = setup_ddp()
    world_size = get_world_size()

    if is_main_process():
        print(f"ActionFormer DDP Training - {world_size} GPUs")

    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if is_main_process():
        pprint(cfg)

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

    rng_generator = fix_random_seed(cfg['init_rand_seed'] + local_rank, include_cuda=True)

    effective_batch_scale = world_size * args.accum_steps
    cfg['opt']["learning_rate"] *= effective_batch_scale
    cfg['loader']['num_workers'] = max(1, cfg['loader']['num_workers'] // world_size)

    train_dataset = make_dataset(cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset'])
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                        rank=dist.get_rank(), shuffle=True, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['loader']['batch_size'], shuffle=False,
        num_workers=cfg['loader']['num_workers'], collate_fn=trivial_batch_collator,
        pin_memory=True, sampler=train_sampler, drop_last=True
    )

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader) // args.accum_steps
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    model_ema = None
    if is_main_process():
        model_ema = ModelEma(model)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
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
        del checkpoint

    if is_main_process():
        with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
            pprint(cfg, stream=fid)

    max_epochs = cfg['opt'].get('early_stop_epochs',
                                 cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])

    for epoch in range(args.start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch_ddp(
            train_loader, model, optimizer, scheduler, epoch, local_rank,
            model_ema=model_ema, clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer, print_freq=args.print_freq,
            use_amp=args.amp, scaler=scaler, accum_steps=args.accum_steps
        )

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
                save_checkpoint(save_states, False, file_folder=ckpt_folder,
                                file_name=f'epoch_{epoch + 1:03d}.pth.tar')
        dist.barrier()

    if is_main_process():
        tb_writer.close()
        print("Training complete!")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='DDP Training for ActionFormer')
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int)
    parser.add_argument('-e', '--eval-freq', default=0, type=int)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--accum-steps', default=1, type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
