from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

from tensorboardX import SummaryWriter       # importing tensorboard
from utils.early_stopping import EarlyStopping
from test import test

from torch.autograd import Variable

import torch.onnx
import torchvision

opt = opts().parse()
Dataset = get_dataset(opt.dataset, opt.task)
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
model = create_model(opt.arch, opt.heads, opt.head_conv)

dummy_input = Variable(torch.randn(1, 3, 511, 511))
state_dict = torch.load('../../pytorch-image-models/help-52ad8d9f.pth')
model.load_state_dict(state_dict)
torch.onnx.export(model, dummy_input, "help.onnx")