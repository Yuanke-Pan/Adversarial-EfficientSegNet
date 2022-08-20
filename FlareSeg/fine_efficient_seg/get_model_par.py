import os
import sys
import time
import argparse
import warnings

import torch
from thop import clever_format
from thop import profile

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from BaseSeg.config.config import get_cfg_defaults
from Common.gpu_utils import set_gpu, run_multiprocessing
from BaseSeg.engine.segmentor_multiprocess import SegmentationMultiProcess
from BaseSeg.network.get_model import get_coarse_model, get_fine_model, get_semi_model


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./config.yaml')

    coarse_model = get_coarse_model(cfg, 'test')
    fine_model = get_fine_model(cfg, 'test')
    semi_model = get_semi_model(cfg, 'test')

    coarse_input = torch.randn(1, 1, 160, 160, 160)
    fine_input = torch.randn(1, 1, 192, 192, 192)

    target_input = torch.randn(1, 13, 160, 160, 160)

    coarse_flops, coarse_params = profile(coarse_model, inputs=(coarse_input,))
    coarse_flops, coarse_params = clever_format([coarse_flops * 2, coarse_params], "%.3f")

    fine_flops, fine_params = profile(fine_model, inputs=(fine_input,))
    fine_flops, fine_params = clever_format([fine_flops * 2, fine_params], "%.3f")

    semi_flops, semi_params = profile(semi_model, inputs=(coarse_input, target_input, target_input))
    semi_flops, semi_params = clever_format([semi_flops * 2, semi_params], "%.3f")

    print("Number of coarse_parameter: ", coarse_params)
    print("Number of coarse_flops: ", coarse_flops)
    print("Number of fine_parameter:",  fine_params)
    print("Number of fine_flops:",  fine_flops)
    print("Number of semi_parameter:", semi_params)
    print("Number of semi_flops:",semi_flops)


