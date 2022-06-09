#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Thu 02 Jun 2022 11:08:31 AM HKT
# @Desc: dataset

import os
import h5py
import torch
import numpy as np

from functools import partial
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from IsoFusion.config import KERNEL_RT, KERNEL_MZ, MAX_CH, MAX_MZ, MAX_RT


class IsoDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.data = []
        self.feats = []
        self.process_num = args.process_num
        with h5py.File(os.path.join(args.output, f"{args.file_name}.hdf5"), 'r') as f:
            self.rt_list = f['rt_list'][:]
            self.rt_search_index = f['rt_list'].attrs['rt_search_index']
            self.min_mz = f['rt_list'].attrs['min_mz']
            self.ms1 = csr_matrix((f['data'][()], f['indices'][()], f['indptr'][()]), shape=f.attrs['shape'])
        self.process_ms1()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.data[idx]).float(), np.array(self.feats[idx])

    def process_ms1(self) -> None:
        _func = partial(traverse_ms1, self.rt_list, self.rt_search_index, self.ms1, self.min_mz)
        with Pool(processes=self.process_num) as pool:
            res = pool.map_async(_func, range(1000, self.ms1.shape[1] - 1000, 50))
            res = res.get()
        for item in res:
            self.data.extend(item[0])
            self.feats.extend(item[1])


def traverse_ms1(rt_list, rt_search_index, ms1, min_mz, mz_idx):
    y = mz_idx
    print(y, end=' ')
    data_list, feat_list = [], []
    while y < mz_idx + 50:
        x = 1000  # since padding exists
        mz = (y - 1000) * 0.01 + min_mz
        ppm = int(round(mz * 1e-5 / 0.01, 2))

        while x < ms1.shape[0] - 1000:
            block = ms1[x: x + 5, y: y + 1].toarray()
            if (block != 0).sum() != 5:
                x += 5
            else:
                x0, x1 = x - KERNEL_RT, x + MAX_RT + KERNEL_RT
                y0, y1 = y - KERNEL_MZ, y + 100 * (MAX_MZ - 1) + KERNEL_MZ + 1
                data, data_lst = ms1[x0: x1, y0: y1].toarray(), []
                for c in range(1, MAX_CH + 1):
                    i = (np.arange(0, 100*MAX_MZ, 100)[:, None] // c + np.arange(KERNEL_MZ*2+1)[None, :]).reshape(-1)
                    d = data[None, None, :, i] / np.max(data[:, i])
                    data_lst.append(d)
                data, data_lst = np.concatenate(data_lst, axis=1), None
                # add data to list
                data_list.append(data)
                # rt
                rt_start = rt_list[rt_search_index + (x - 1000)]
                _data = ms1[x: x + MAX_RT // 2, y - 1: y + 1]
                rt_apex_idx = x + np.unravel_index(_data.argmax(), _data.shape)[0]
                rt_apex = rt_list[rt_apex_idx + rt_search_index - 1000]
                feat_list.append((mz, rt_start, rt_apex, x, y))

                x += MAX_RT
        y += (ppm + 1)
    return data_list, feat_list