#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import numpy as np
import os
from datetime import datetime
import psutil
import torch
from fvcore.nn.flop_count import flop_count
from matplotlib import pyplot as plt
from torch import nn

from . import logging

logger = logging.get_logger(__name__)


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.
    Returns:
        float: the total number of gflops of the given model.
    """
    rgb_dimension = 3
    input_tensors = torch.rand(
            rgb_dimension*cfg.channels,
            cfg.image_H,
            cfg.image_W,
        )
    flop_inputs = [input_tensors]
    for i in range(len(flop_inputs)):
        flop_inputs[i] = flop_inputs[i].unsqueeze(0).cuda(non_blocking=True)

    inputs = (flop_inputs[0],)
    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops


def log_model_info(model, cfg, is_train=True):
    """
    Log info, includes number of parameters, gpu usage and gflops.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info(
        "FLOPs: {:,} GFLOPs".format(get_flop_stats(model, cfg, is_train))
    )

def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH
