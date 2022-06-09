#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Wed 23 Feb 2022 08:57:31 PM HKT
# @Desc: preprocess

import os
import h5py
import numpy as np

from collections import defaultdict
from multiprocessing.pool import ThreadPool
from scipy.sparse import csr_matrix


def process_one_file(args):
    """parser .ms1 file, and generate numpy array (rt, mz, I)"""
    rt_mz_I_dict = defaultdict(list)
    print('Loading MS1 data...')
    with open(args.file, 'r') as f:
        for line in f:
            if 'RTime' in line: rt_val = round(float(line.split()[-1]), 2)
            if not line.startswith('H') and not line.startswith('I') and not line.startswith('S'):
                item = line.split()
                mz_val = round(float(item[0]), 3)
                intensity_val = round(float(item[1]), 4)
                rt_mz_I_dict[rt_val].append((mz_val, intensity_val))
    print('MS1 format data loaded!')

    rt_list = sorted(list(rt_mz_I_dict.keys()))
    mz_I_list = [sorted(rt_mz_I_dict[rt]) for rt in rt_list]
    max_mz = max([mz[-1][0] for mz in mz_I_list])
    min_mz = min([mz[0][0] for mz in mz_I_list])
    min_rt, max_rt = 10, rt_list[-1]
    rt_search_index = 0
    while rt_list[rt_search_index] < min_rt:
        rt_search_index += 1
    total_mz = int(round((max_mz - min_mz + 0.01) / 0.01, 3))
    total_rt = len(rt_list) - rt_search_index
    ms1 = np.zeros((total_rt, total_mz), dtype=np.float32)

    print('Filling the numpy array')
    def _func(idx):
        mzI = mz_I_list[idx + rt_search_index]
        temp_dict = defaultdict(lambda: 0)
        for (mz, I) in mzI:
            if mz > max_mz:
                break
            temp_dict[mz] = max(I, temp_dict[mz])
        for mz, I in temp_dict.items():
            mz_idx = int(round((mz - min_mz) / 0.01, 2))
            ms1[idx, mz_idx] = I

    with ThreadPool(processes=6) as pool:
        pool.map_async(_func, range(total_rt)).get()

    # padding
    ms1 = np.concatenate([ms1, np.zeros((1000, ms1.shape[1]))], axis=0)
    ms1 = np.concatenate([np.zeros((1000, ms1.shape[1])), ms1], axis=0)
    ms1 = np.concatenate([ms1, np.zeros((ms1.shape[0], 1000))], axis=1)
    ms1 = np.concatenate([np.zeros((ms1.shape[0], 1000)), ms1], axis=1)
    # sparse
    ms1 = csr_matrix(ms1)

    print('Writing to the disk')
    with h5py.File(os.path.join(args.output, f'{args.file_name}.hdf5'), 'w') as f:
        _h5rt = f.create_dataset('rt_list', data=np.array(rt_list), dtype=np.float32)
        _h5rt.attrs['rt_search_index'] = rt_search_index
        _h5rt.attrs['min_mz'] = min_mz
        _h5rt.attrs['max_mz'] = max_mz
        f.create_dataset('data', data=ms1.data)
        f.create_dataset('indptr', data=ms1.indptr)
        f.create_dataset('indices', data=ms1.indices)
        f.attrs['shape'] = ms1.shape
    print(f"Preprocess of {args.file_name} done!")