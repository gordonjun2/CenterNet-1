from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net

from .networks.large_hourglass_all_dense import get_large_hourglass_net_all_dense
from .networks.large_hourglass_resdense import get_large_hourglass_net_resdense
from .networks.large_hourglass_skip3 import get_large_hourglass_net_skip3
from .networks.large_hourglass_skip_dense import get_large_hourglass_net_skip_dense
from .networks.large_hourglass_resdense_skip_none import get_large_hourglass_net_resdense_skip_none
from .networks.large_hourglass_linked_v1 import get_large_hourglass_net_linked_v1
from .networks.large_hourglass_linked_v2 import get_large_hourglass_net_linked_v2
from .networks.large_hourglass_linked_reduced import get_large_hourglass_net_linked_reduced

from .networks.small_hourglass import get_small_hourglass_net
from .networks.small_hourglass_all_dense import get_small_hourglass_net_all_dense
from .networks.small_hourglass_resdense import get_small_hourglass_net_resdense
from .networks.small_hourglass_skip3 import get_small_hourglass_net_skip3
from .networks.small_hourglass_skip_dense import get_small_hourglass_net_skip_dense
from .networks.small_hourglass_resdense_skip_none import get_small_hourglass_net_resdense_skip_none

_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'resdcn': get_pose_net_dcn,
  'hourglass-104': get_large_hourglass_net,
  'hourglass-52': get_small_hourglass_net,
  'hourglass-104-skip3': get_large_hourglass_net_skip3,
  'hourglass-52-skip3': get_small_hourglass_net_skip3,
  'hourglass-104-skip-dense': get_large_hourglass_net_skip_dense,
  'hourglass-52-skip-dense': get_small_hourglass_net_skip_dense,
  'hourglass-104-resdense': get_large_hourglass_net_resdense,
  'hourglass-52-resdense': get_small_hourglass_net_resdense,
  'hourglass-104-all-dense': get_large_hourglass_net_all_dense,
  'hourglass-52-all-dense': get_small_hourglass_net_all_dense,
  'hourglass-104-resdense-skip-none': get_large_hourglass_net_resdense_skip_none,
  'hourglass-52-resdense-skip-none': get_small_hourglass_net_resdense_skip_none,
  'hourglass-104-linked-v1': get_large_hourglass_net_linked_v1,
  'hourglass-104-linked-v2': get_large_hourglass_net_linked_v2,
  'hourglass-104-linked-reduced': get_large_hourglass_net_linked_reduced
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

