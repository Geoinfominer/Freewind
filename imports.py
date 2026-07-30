import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import re
import copy
import math
from torch.nn.parameter import Parameter
from tqdm import trange
import pytorch_lightning as pl # verson 2.1.3
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import callbacks
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import dask
dask.config.set(scheduler="threads")
from torch.profiler import record_function
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import schedule, tensorboard_trace_handler
import torch.distributed as dist
import time