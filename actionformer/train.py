#!/usr/bin/env python3
"""
ActionFormer single-GPU training CLI.

Usage:
    actionformer-train configs/thumos_i3d.yaml --output my_exp
"""

import argparse
import os
import time
import datetime
from pprint import pprint

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (
    train_one_epoch, valid_one_epoch, ANETdetection,
    save_checkpoint, make_optimizer, make_scheduler,
    fix_random_seed, ModelEma
)


def run(args):
    """Main training function."""
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    print("Using model EMA ...")
    model_ema = ModelEma(model)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(
                args.resume,
                map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
            )
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    print("\nStart training model {:s} ...".format(cfg['model_name']))

    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict_ema': model_ema.module.state_dict(),
            }
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    tb_writer.close()
    print("All done!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train ActionFormer for action localization (single GPU)'
    )
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
    run(args)


if __name__ == '__main__':
    main()
