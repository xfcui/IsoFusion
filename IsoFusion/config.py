#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Thu 02 Jun 2022 11:08:31 AM HKT
# @Desc: configure parameters

MAX_CH = 9  # covers 100%
MAX_RT = 20 # covers 95%
MAX_MZ = 5  # covers 95%

KERNEL_RT = 3
KERNEL_MZ = 2

MODEL_FUSE   = 2
MODEL_EVAL   = 1

MODEL_SAV = 'IsoFusion/model_sav/iso1201-epoch0020.pt'
THRESHOLD = {1: 0.88, 2: 0.73, 3: 0.74, 4: 0.75, 5: 0.85, 6: 0.85, 7: 0.6, 8: 0.1, 9: 0.1}