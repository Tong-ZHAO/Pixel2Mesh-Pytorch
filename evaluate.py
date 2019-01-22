import os, sys
sys.path.append('data')
sys.path.append('model')
sys.path.append('utils')

import numpy as np
from utils import *
from model import *
from loss import *

import torch
import torch.optim as optim
import argparse
import time, datetime
import random


# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type = str, default = '/home/pkq/tong/MVA/S1/objet/project/data/ShapeNetSmall/02691156_d3b9114df1d8a3388e415c6cf89025f0_02.dat', 
                    help = 'the path to find the data')
parser.add_argument('--modelPath', type = str, default = 'log/2019-01-18T02:32:24.320546/network_4.pth',  help = 'the path to find the trained model')
parser.add_argument('--savePath', type = str, default = 'eval/',  help = 'the path to save the reconstructed meshes')
parser.add_argument('--saveName', type = str, default = 'out_mesh',  help = 'the name of the output mesh')
parser.add_argument('--offPath', type = str, default = 'data/ellipsoid/',  help = 'the path to save the mesh surface')
parser.add_argument('--hidden', type = int, default = 192,  help = 'number of units in  hidden layer')
parser.add_argument('--featDim', type = int, default = 963,  help = 'number of units in perceptual feature layer')
parser.add_argument('--coordDim', type = int, default = 3,  help='number of units in output layer')

opt = parser.parse_args()
print (opt)

# Check Device (CPU / GPU)
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

ellipsoid = read_init_mesh('data/info_ellipsoid.dat')

# Create Network
network = P2M_Model(opt.featDim, opt.hidden, opt.coordDim, ellipsoid['pool_idx'], ellipsoid['supports'], use_cuda)
#network.apply(weights_init) #initialization of the weight
network.load_state_dict(torch.load(opt.modelPath))
network.eval()

if use_cuda:
    network.cuda()

def load_file(file_path):

    data = pickle.load(open(file_path, "rb"), encoding = 'latin1')
    img, pts, normals = data[0].astype('float32') / 255.0, data[1][:, :3], data[1][:, 3:]
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis = 0)

    return torch.from_numpy(img), torch.from_numpy(pts), normals

img, pts, _ = load_file(opt.dataPath)
init_pts = torch.from_numpy(ellipsoid['coord'])

if use_cuda:
    img = img.cuda()
    pts = pts.cuda()
    init_pts = init_pts.cuda()

pred_pts_list, pred_feats_list, pred_img = network(img, init_pts)

for i in range(len(pred_pts_list)):
    face = np.loadtxt(os.path.join(opt.offPath, "face" + str(i + 1) + ".obj"), dtype = '|S32')
    vert = pred_pts_list[i].cpu().data.numpy()
    vert = np.hstack((np.full([vert.shape[0], 1], 'v'), vert))
    mesh = np.vstack((vert, face))
    np.savetxt(os.path.join(opt.savePath, opt.saveName + "_" + str(i + 1) + ".obj"), mesh, fmt = '%s', delimiter = ' ')

np.savetxt(os.path.join(opt.savePath, opt.saveName + "_gt.xyz"), pts.cpu().data.numpy(), fmt = '%s', delimiter = ' ')

print("Finish!")