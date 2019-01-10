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

    def __init__(self, features_dim, hidden_dim, coord_dim, pool_idx, supports, use_cuda):

        super(P2M_Model, self).__init__()
        self.img_size = 224

        self.features_dim = features_dim
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.pool_idx = pool_idx
        self.supports = supports
        self.use_cuda = use_cuda

        self.build()


    def build(self):

        self.nn_encoder = self.build_encoder()
        self.nn_decoder = self.build_decoder()

        self.GCN_0 = GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim, self.supports[0], self.use_cuda)
        self.GCN_1 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim, self.supports[1], self.use_cuda)
        self.GCN_2 = GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.hidden_dim, self.supports[2], self.use_cuda)

        self.GPL_1 = GraphPooling(self.pool_idx[0])
        self.GPL_2 = GraphPooling(self.pool_idx[1])

        self.GPR_0 = GraphProjection() 
        self.GPR_1 = GraphProjection()
        self.GPR_2 = GraphProjection()

        self.GConv = GraphConvolution(in_features = self.hidden_dim, out_features = self.coord_dim, adjs = self.supports[2], use_cuda = self.use_cuda)


    def forward(self, img, input):

        img_feats = self.nn_encoder(img)

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
        x, _ = self.GCN_2(x)

        x = self.GConv(x)

        new_img = self.nn_decoder(img_feats)

        return x, new_img


    def build_encoder(self):
        # VGG16 at first, then try resnet
        # Can load params from model zoo
        net = VGG16_Pixel2Mesh(n_classes_input = 3)
        return net

    def build_decoder(self):
        net = VGG16_Decoder()
        return net


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, adjs, use_cuda):
        super(GResBlock, self).__init__()

        self.conv1 = GraphConvolution(in_features = in_dim, out_features = hidden_dim, adjs = adjs, use_cuda = use_cuda)
        self.conv2 = GraphConvolution(in_features = in_dim, out_features = hidden_dim, adjs = adjs, use_cuda = use_cuda) 


    def forward(self, input):
        
        x = self.conv1(input)
        x = self.conv2(x)

        return (input + x) * 0.5


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, adjs, use_cuda):
        super(GBottleneck, self).__init__()

        blocks = [GResBlock(in_dim = hidden_dim, hidden_dim = hidden_dim, adjs = adjs, use_cuda = use_cuda)]

        for _ in range(block_num - 1):
            blocks.append(GResBlock(in_dim = hidden_dim, hidden_dim = hidden_dim, adjs = adjs, use_cuda = use_cuda))

        self.blocks = nn.Sequential(*blocks)
        self.conv1 = GraphConvolution(in_features = in_dim, out_features = hidden_dim, adjs = adjs, use_cuda = use_cuda)
        self.conv2 = GraphConvolution(in_features = hidden_dim, out_features = out_dim, adjs = adjs, use_cuda = use_cuda)

        
    def forward(self, input):

        x = self.conv1(input)
        x_cat = self.blocks(x)
        x_out = self.conv2(x_cat)

        return x_out, x_cat
    
    
class VGG16_Pixel2Mesh(nn.Module):
    
    def __init__(self, n_classes_input = 3):
        
        super(VGG16_Pixel2Mesh, self).__init__()

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride = 1, padding = 1)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1)
        
        self.conv1_1 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1) # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1) # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1) # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        
        self.conv4_1 = nn.Conv2d(128, 256, 5, stride = 2, padding = 2) # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
        
        self.conv5_1 = nn.Conv2d(256, 512, 5, stride = 2, padding = 2) # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        
    def forward(self, img):
        
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))
        #img0 = torch.squeeze(img) # 224
        
        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))
        #img1 = torch.squeeze(img) # 112 
        
        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = torch.squeeze(img) # 56
        
        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img3 = torch.squeeze(img) # 28
        
        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img4 = torch.squeeze(img) # 14
        
        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = torch.squeeze(img) #7
        
        return [img2, img3, img4, img5]


class VGG16_Decoder(nn.Module):

    def __init__(self, input_dim = 512, image_channel = 3):

        super(VGG16_Decoder, self).__init__()

        self.conv_1 = nn.ConvTranspose2d(input_dim, 256, kernel_size = 2, stride = 2, padding = 0) # 7 -> 14
        self.conv_2 = nn.ConvTranspose2d(512, 128, kernel_size = 4, stride = 2, padding = 1) # 14 -> 28
        self.conv_3 = nn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 2, padding = 1)  # 28 -> 56
        self.conv_4 = nn.ConvTranspose2d(128, 32, kernel_size = 6, stride = 2, padding = 2)   # 56 -> 112
        self.conv_5 = nn.ConvTranspose2d(32, image_channel, kernel_size = 6, stride = 2, padding = 2) # 112 -> 224


    def forward(self, img_feats):

        x = F.relu(self.conv_1(img_feats[-1].unsqueeze(0)))
        x = torch.cat((x, img_feats[-2].unsqueeze(0)), dim = 1)
        x = F.relu(self.conv_2(x))
        x = torch.cat((x, img_feats[-3].unsqueeze(0)), dim = 1)
        x = F.relu(self.conv_3(x))
        x = torch.cat((x, img_feats[-4].unsqueeze(0)), dim = 1)
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))

        return F.sigmoid(x)


