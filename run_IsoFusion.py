#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Mon 04 Apr 2022 09:06:01 PM HKT
# @Desc: main

import os
import torch
import pickle
import argparse

from torch.utils.data import DataLoader

from IsoFusion.data import IsoDataset
from IsoFusion.model import IsoFusion
from IsoFusion.preprocess import process_one_file
from IsoFusion.config import MODEL_EVAL, MODEL_FUSE, MODEL_SAV, THRESHOLD


def run(args):
    # preprocess
    process_one_file(args)

    print('Loading model...')
    width, depth = 32*4, 6
    model = IsoFusion(width, depth, width_scale=4, num_group=4, depth_scale=2, level_fuse=MODEL_FUSE)
    model.load_state_dict(torch.load(MODEL_SAV, map_location=lambda storage, loc: storage))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print('Model loaded')

    print('Loading dataset...')
    dataset = IsoDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    print('Dataset loaded')

    # process
    print('Model processing...')
    result = []
    with torch.no_grad():
        for data in loader:
            inputs, feats = data
            feats = feats.numpy()
            if MODEL_EVAL > 0:  # hard label
                pred_ch, pred_rt, pred_iso = model(inputs.to(device), 1)
            else:               # soft label
                pred_ch, pred_rt, pred_iso = model(inputs.to(device))

            arg_max_ch = pred_ch.detach().cpu().argmax(dim=-1).numpy() + 1
            arg_max_iso = pred_iso.detach().cpu().argmax(dim=-1).numpy() + 1
            arg_max_rt = pred_rt.detach().cpu().argmax(dim=-1).numpy() + 1

            # remove noise
            soft_ch = torch.nn.functional.softmax(pred_ch, dim=1).detach().cpu()
            soft_ch = soft_ch.max(1)[0]
            con = []
            for ch, soft in zip(arg_max_ch, soft_ch):
                con.append(soft >= THRESHOLD[ch])
            feats = feats[con]
            arg_max_ch = arg_max_ch[con]
            arg_max_rt = arg_max_rt[con]
            arg_max_iso = arg_max_iso[con]

            for idx, val in enumerate(arg_max_ch):
                result.append((*feats[idx], arg_max_rt[idx], val, arg_max_iso[idx]))
    
    with open(os.path.join(args.output, args.file_name + '_report.pkl'), 'wb') as f:
        pickle.dump(result, f)
    print('Done!')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='the target file (absolute path)')
    parser.add_argument('--output', type=str, default='output', help='the dir where results will be saved')
    parser.add_argument('--process_num', type=int, default=8, help='multiprocess')
    parser.add_argument('--gpu', type=int, default=0, help='specify the gpu num you use')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size will be used')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    args.file_name = os.path.basename(args.file).split('.')[0]
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    run(args)
