# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

import math
from matplotlib import pyplot as plt

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_up_layer(modules, dim):
    growth_rate = dim / 2
    #nb_layers = 1           # Hyperparameters, but treat as a constant first
    reduction = 0.5         # Hyperparameters, but treat as a constant first
    dropRate = 0            # Hyperparameters, but treat as a constant first
    depth = 10
    in_planes = 2 * growth_rate
    n = (depth - 4) / 3
    n = int(n)
    layers = []

    for _ in range(modules):
        for i in range(n):
            layers.append(DenseBlock(nb_layers = i, in_planes = in_planes, growth_rate = growth_rate, block = BasicBlock, dropRate = dropRate))  
        in_planes = int(in_planes+n*growth_rate)  
        layers.append(TransitionBlock(in_planes = in_planes, out_planes = int(math.floor(in_planes*reduction)), dropRate = dropRate))
        in_planes = int(math.floor(in_planes*reduction))
    
    return nn.Sequential(*layers)

def make_hg_layer(kernel, mod, dim0, dim1, nDenselayer):
    #nDenselayer = 2             # Hyperparameters, but treat as a constant first
    growthRate = dim0 // 2        # Hyperparameters, but treat as a constant first
    nChannels = int(growthRate * 2)
    layers = []
    nChannels_def = nChannels
    nChannels_ = nChannels

    for _ in range(mod):
        for i in range(nDenselayer):
            layers.append(RDB(nChannels = nChannels_, nDenselayer = nDenselayer, growthRate = growthRate))
            nChannels_ += growthRate
        layers.append(nn.Conv2d(nChannels_ - growthRate, nChannels_def, kernel_size=1, padding=0, bias=False))
        layers.append(nn.Conv2d(nChannels_def, dim1, kernel_size=1, padding=0, bias=True))
        layers.append(nn.Conv2d(dim1, dim1, kernel_size=3, padding=1, bias=True))         # global feature fusion (GFF)
        nChannels_ = dim1

    return nn.Sequential(*layers)

# def make_hg_layer_revr(kernel, mod, dim0, dim1, nDenselayer):
#     #nDenselayer = 2             # Hyperparameters, but treat as a constant first
#     growthRate = dim0 // 2        # Hyperparameters, but treat as a constant first
#     nChannels = int(growthRate * 2)
#     layers = []
#     nChannels_def = nChannels
#     nChannels_ = nChannels

#     for _ in range(mod):
#         for i in range(nDenselayer):
#             layers.append(RDB(nChannels = nChannels_, nDenselayer = nDenselayer, growthRate = growthRate))
#             nChannels_ += growthRate
#         #layers.append(nn.Conv2d(nChannels_, nChannels_def, kernel_size=1, padding=0, bias=False))
#         #layers.append(nn.Conv2d(nChannels_def, dim1, kernel_size=1, padding=0, bias=True))
#         layers.append(nn.Conv2d(nChannels_, dim1, kernel_size=3, padding=1, bias=True))         # global feature fusion (GFF)
#         nChannels_ = dim1

#     return nn.Sequential(*layers) 

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(int(nChannels), int(growthRate), kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)

    def forward(self, x):
        out = nn.functional.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        self.layer = self._make_layer(nChannels, nDenselayer, growthRate)
    def _make_layer(self, nChannels, nDenselayer, growthRate):
        #modules = []
        #nChannels_def = nChannels
        #nChannels_ = nChannels
        #for i in range(nDenselayer):    
        module = make_dense(nChannels, growthRate)

        #modules.append(nn.Conv2d(nChannels_, nChannels_def, kernel_size=1, padding=0, bias=False))
        #self.dense_layers = nn.Sequential(*modules)    
        #self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
        return module

    def forward(self, x):
        #out = self.dense_layers(x)
        #out = self.conv_1x1(out)
        #out = out + x
        return self.layer(x)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_hg_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  

        self.max1 = make_pool_layer(curr_dim)
        #self.low1 = make_hg_layer(
        #    3, curr_dim, next_dim, curr_mod,
        #    layer=layer, **kwargs
        #)

        self.low1 = make_hg_layer(kernel = 3, mod = curr_mod, dim0 = curr_dim, dim1 = next_dim, nDenselayer = 1)                  # RDB here

        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )

        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )   

        #self.low3 = make_hg_layer_revr(kernel = 3, mod = curr_mod, dim0 = next_dim, dim1 = curr_dim, nDenselayer = 1)                  # RDB here

        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)

        ##### See Feature Map Here #####
        # feature_map = max1.cpu().detach().numpy()
        # print("max1 Shape:", feature_map.shape)
        # feature_map = np.sum(feature_map[0], axis = 0)
        # plt.imshow(feature_map/feature_map[1])
        # plt.show()
        ################################

        low1 = self.low1(max1)

        ##### See Feature Map Here #####
        # feature_map = low1.cpu().detach().numpy()
        # print("low1 Shape:", feature_map.shape)
        # feature_map = np.sum(feature_map[0], axis = 0)
        # plt.imshow(feature_map/feature_map[1])
        # plt.show()
        ################################

        low2 = self.low2(low1)

        ##### See Feature Map Here #####
        # feature_map = low2.cpu().detach().numpy()
        # print("low2 Shape:", feature_map.shape)
        # feature_map = np.sum(feature_map[0], axis = 0)
        # plt.imshow(feature_map/feature_map[1])
        # plt.show()
        ################################

        low3 = self.low3(low2)

        ##### See Feature Map Here #####
        # feature_map = low3.cpu().detach().numpy()
        # print("low3 Shape:", feature_map.shape)
        # feature_map = np.sum(feature_map[0], axis = 0)
        # plt.imshow(feature_map/feature_map[1])
        # plt.show()
        ################################

        up2  = self.up2(low3)

        ##### See Feature Map Here #####
        # feature_map = up2.cpu().detach().numpy()
        # print("up2 Shape:", feature_map.shape)
        # feature_map = np.sum(feature_map[0], axis = 0)
        # plt.imshow(feature_map/feature_map[1])
        # plt.show()
        ################################

        ##### See Feature Map Here #####
        # feature_map = self.merge(up1, up2).cpu().detach().numpy()
        # print("merged Shape:", feature_map.shape)
        # feature_map = np.sum(feature_map[0], axis = 0)
        # plt.imshow(feature_map/feature_map[1])
        # plt.show()
        ################################
        
        return self.merge(up1, up2)

class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_hg_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        # print('image shape', image.shape)

        ##### See Feature Map Here #####
        # input = image.cpu().detach().numpy()
        # input = np.sum(input[0], axis = 0)
        # plt.imshow(input/input[1])
        # plt.show()
        ################################

        inter = self.pre(image)
        outs  = []

        for ind in range(self.nstack):
            kp_, cnv_  = self.kps[ind], self.cnvs[ind]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out[head] = y
            
            outs.append(out)
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs


def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=1):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

def get_small_hourglass_net_resdense(num_layers, heads, head_conv):
  model = HourglassNet(heads, 1)
  return model
