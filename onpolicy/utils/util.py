import numpy as np
import math
import torch
import random
import socket
from os import path as osp

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

import datetime


def eta(start_time: float, end_time: float, total: int, processed: int) -> str:
    """Compute the estimated time of arrival.

    Args:
        start_time (float): Start time.
        end_time (float): End time.
        total (int): Total number of items.
        processed (int): Number of processed items.

    Returns:
        str: Estimated time of arrival.
    """
    elapsed = end_time - start_time
    eta = (total - processed) * elapsed / processed
    return str(datetime.timedelta(seconds=int(eta)))


def get_table_str(items: list, headers: list = None, title: str = None, sort: bool = True) -> str:
    """
    return a table str
    """
    import io

    from rich.console import Console
    from rich.table import Table

    if headers:
        assert len(headers) == len(items[0])
        table = Table(title=title)
        for h in headers:
            table.add_column(h)
    else:
        table = Table(show_header=False, title=title)
        for _ in range(len(items[0])):
            table.add_column(justify="left")

    if sort:
        items = sorted(items, key=lambda x: x[0])

    for v in items:
        table.add_row(*[str(_v) for _v in v])
    string_io = io.StringIO()
    console = Console(file=string_io, record=True)
    console.print(table)
    return console.export_text()

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_base_run_dir() -> str:
    socket.gethostname()
    base = osp.join(osp.expanduser("~"), "ZSC/results")
    return base