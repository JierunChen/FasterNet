import torch
from torch import nn
import os
import numpy as np
import yaml
import time
from argparse import Namespace
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_only
from fvcore.nn import FlopCountAnalysis, parameter_count


def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return Namespace(**hyp)


def merge_args_cfg(args, cfg):
    dict0 = vars(args)
    dict1 = vars(cfg)
    dict = {**dict0, **dict1}

    return Namespace(**dict)


def str2list(string, sperator=','):
    li = list(map(int, string.split(sperator)))
    return li

# def check_dir_format(dir):
#     if dir.endswith(os.path.sep):
#         return dir
#     else:
#         return dir+os.path.sep
#
#
#
#
#
# def append_path_by_date(model_ckpt_dir):
#     os.environ['TZ'] = 'Asia/Hong_Kong'
#     time.tzset()
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#
#     return check_dir_format(model_ckpt_dir) + timestr + os.path.sep
#
@torch.no_grad()
@rank_zero_only
def print_model(model):
    print(model)


@torch.no_grad()
def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new())

# @torch.no_grad()
# @rank_zero_only
# def mk_model_ckpt_dir(model_ckpt_dir):
#     if not os.path.exists(model_ckpt_dir):
#         os.makedirs(model_ckpt_dir)
#
@torch.no_grad()
def get_flops_params(model, input_size):
    model.eval()

    tensor = (torch.rand(1, 3, input_size, input_size), )
    flops = FlopCountAnalysis(model, tensor)
    flops = flops.total() / 1000000.
    print("FVcore FLOPs(M): ", flops)

    params = parameter_count(model)
    params = params[""] / 1000000.
    print("FVcore params(M): ", params)

    return flops, params


@torch.no_grad()
def measure_latency(images, model, GPU=True, chan_last=False, half=False, num_threads=None, iter=200):
    """
    :param images: b, c, h, w
    :param model: model
    :param GPU: whther use GPU
    :param chan_last: data_format
    :param half: half precision
    :param num_threads: for cpu
    :return:
    """

    if GPU:
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True

        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()

        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(iter):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on gpu {throughput}")
        print(f"batch_size {batch_size} latency on gpu {latency} ms")

        return throughput, latency
    else:
        model.eval()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        batch_size = images.shape[0]

        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()
        for i in range(10):
            model(images)
        tic1 = time.time()
        for i in range(iter):
            model(images)
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on cpu {throughput}")
        print(f"batch_size {batch_size} latency on cpu {latency} ms")

        return throughput, latency
#
#
# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print
#
#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)
#
#     __builtin__.print = print
#
#
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()