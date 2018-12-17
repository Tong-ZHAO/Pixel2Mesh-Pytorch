import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from gcn_layers import *

class P2M_Model(nn.Module):
    """
    Implement the joint model for Pixel2mesh
    """

    def __init__(self, use_cuda = False, features_dim, hidden_dim, coord_dim):

        super(P2M_Model, self).__init__()
        self.use_cuda = use_cuda
        self.img_size = 224

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim


    def build(self):

        self.2Dnn = self.build_2dnn()

        self.GCN_0 = GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim)
        self.GCN_1 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim)
        self.GCN_2 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.hidden_dim)

        self.GPL_1 = GraphPooling(self.pool_idx[0])
        self.GPL_2 = GraphPooling(self.pool_idx[1])

        self.GPR_0 = GraphProjection() 
        self.GPR_1 = GraphProjection()
        self.GPR_2 = GraphProjection()

        self.GConv = GraphConvolution(n_features = self.hidden_dim, out_features = self.coord_dim)


    def forward(self, img, input):

        img_feats = self.2Dnn(img)

        # GCN Block 1
        x = self.GPR_0(img_feats, input)
        x, x_cat = self.GCN_0(x)

        # GCN Block 2
        x = self.GPR_1(img_feats, x)
        x = torch.cat([x, x_cat], 1)
        x = self.GPL_1(x)
        x, x_cat = self.GCN_1(x)
        
        # GCN Block 3
        x = self.GPR_2(img_feats, x)
        x = torch.cat([x, x_cat], 1)
        x = self.GPL_2(x)
        x = self.GCN_2(x)

        x = self.GConv(x)

        return x


    def build_2dnn(self):
        # VGG16 at first, then try resnet
        # Can load params from model zoo
        return net


class GResBlock(nn.Module):

    def __init__(self, hidden_dim):
        super(GResBlock, self).__init__()

        self.conv1 = GraphConvolution(in_features = hidden_dim, out_features = hidden_dim)
        self.conv2 = GraphConvolution(in_features = hidden_dim, out_features = hidden_dim) 


    def forward(self, input):
        
        x = self.conv1(input)
        x = self.conv2(x)

        return (input + x) * 0.5


class GBottleneck(nn.Module):

    def __init(self, block_num, in_dim, hidden_dim, out_dim):
        super(GBottleneck, self).__init__()

        blocks = [GResBlock(in_dim = in_dim, hidden_dim = hidden_dim)]

        for _ in range(block_num - 1):
            blocks.append(GResBlock(in_dim = hidden_dim, hidden_dim = hidden_dim))

        self.blocks = nn.Sequential(*blocks)
        self.conv1 = GraphConvolution(in_features = in_dim, out_features = hidden_dim)
        self.conv2 = GraphConvolution(in_features = hidden_dim, out_features = out_dim)

        
    def forward(self, input):

        x = self.conv1(input)
        x_cat = self.blocks(x)
        x_out = self.conv2(x_cat)

        return x_out, x_cat