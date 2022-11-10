"""
STG-NF modules, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch
"""


import math
import torch


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def split_feature(tensor, type="split", imgs=False):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if imgs:
        if type == "split":
            return tensor[:, : C // 2, ...], tensor[:, C // 2:, ...]
        elif type == "cross":
            return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

    if type == "split":
        return tensor[:, : C // 2, ...].squeeze(dim=1), tensor[:, C // 2 :, ...].squeeze(dim=1)
    elif type == "cross":
        return tensor[:, 0::2, ...].squeeze(dim=1), tensor[:, 1::2, ...].squeeze(dim=1)
