#!/usr/bin/env python3
"""
ActionFormer evaluation CLI.

Usage:
    actionformer-eval configs/thumos_i3d.yaml ./ckpt/my_exp
"""

import argparse
import os
import glob
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


def run(args):
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch))
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    _ = fix_random_seed(0, include_cuda=True)

    val_dataset = make_dataset(cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    print("=> loading checkpoint '{}'".format(ckpt_file))
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader, model, -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))


def main():
    parser = argparse.ArgumentParser(description='Evaluate ActionFormer model')
    parser.add_argument('config', type=str, metavar='DIR', help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR', help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1, help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions')
    parser.add_argument('--saveonly', action='store_true', help='Only save outputs without evaluation')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
