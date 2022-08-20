import numpy as np
import torch
import torch.nn.functional as F
import math

def mask_deterioration(input, cfg, one_hot_flag=False):
    one_hot = input * math.exp(1)
    # wait to add distribute learning
    random1 = torch.rand(one_hot.shape, device='cuda') * 0.5 + 0.5
    random2 = torch.rand(one_hot.shape, device='cuda') * 0.5 + 0.5
    output = F.softmax((one_hot + random1) * random2, dim=-1)
    return output