import os, sys
sys.path.append('data')
sys.path.append('model')
sys.path.append('utils')

import numpy as np
from utils import *
from ShapeNet import *
from model import *
from loss import *

import torch
import torch.optim as optim
import argparse
import time, datetime
import visdom
import random


# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type = str, default = '../data/ShapeNetSmall/', help = 'file root')
parser.add_argument('--dataTrainList', type = str, default = 'data/train_list_plane.txt', help = 'train file list')
parser.add_argument('--dataTestList', type = str, default = 'data/test_list_plane.txt', help = 'test file list')
parser.add_argument('--workers', type = int, help = 'number of data loading workers', default = 12)
parser.add_argument('--nEpoch', type = int, default = 30, help = 'number of epochs to train for')
parser.add_argument('--hidden', type = int, default = 192,  help = 'number of units in  hidden layer')
parser.add_argument('--featDim', type = int, default = 963,  help = 'number of units in perceptual feature layer')
parser.add_argument('--coordDim', type = int, default = 3,  help='number of units in output layer')
parser.add_argument('--weightDecay', type = float, default = 5e-6, help = 'weight decay for L2 loss')
parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
parser.add_argument('--env', type = str, default = "pixel2mesh", help = 'visdom environment')
parser.add_argument('--lamb', type = float, default = 0.0001, help = 'loss coeff for img reconstruction task')

opt = parser.parse_args()
print (opt)

# Read initial mesh
num_blocks = 3
num_supports = 2
ellipsoid = read_init_mesh('data/info_ellipsoid.dat')

# Check Device (CPU / GPU)
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



# Launch visdom for visualization
vis = visdom.Visdom(port = 8097, env = opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name =  os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10

# Create Dataset
dataset = ShapeNet(opt.dataRoot, opt.dataTrainList)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers=int(opt.workers))
dataset_test = ShapeNet(opt.dataRoot, opt.dataTestList)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle = False, num_workers = int(opt.workers))
len_dataset = len(dataset)
print('training set', len_dataset)
print('testing set', len(dataset_test))


# Create Network
network = P2M_Model(opt.featDim, opt.hidden, opt.coordDim, ellipsoid['pool_idx'], ellipsoid['supports'], use_cuda)
network.apply(weights_init) #initialization of the weight

if use_cuda:
    network.cuda()

# Create Optimizer
lrate = opt.lr
optimizer = optim.Adam(network.parameters(), lr = lrate)

# meters to record stats on learning
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
    f.write(str(network) + '\n')

# Initialize visdom
vis_title = 'Pixel2Mesh'
vis_legend = ['Image Loss', 'Mesh Loss', 'Total Loss']
iter_plot = create_vis_plot(vis, 'Iteration', 'Loss', vis_title, vis_legend)



# Train model on the dataset
for epoch in range(opt.nEpoch):

    # Initialize loss
    my_img_loss = 0.
    my_pts_loss = 0.

    update_vis_plot(vis, 0, my_img_loss, my_pts_loss, iter_plot, "replace")
    # Set to Train mode
    train_loss.reset()
    network.train()
    # learning rate schedule
    if epoch == 10:
        lrate = lrate / 10.
        optimizer = optim.Adam(network.parameters(), lr = lrate)

    for i, data in enumerate(dataloader, 0):

        if i != 0 and (i % 50 == 0):
            update_vis_plot(vis, i, my_img_loss, my_pts_loss, iter_plot, "append")

        optimizer.zero_grad()
        img, pts, normal, _, _ = data
        init_pts = torch.from_numpy(ellipsoid['coord'])

        if use_cuda:
            img = img.cuda()
            pts = pts.cuda()
            normal = normal.cuda()
            init_pts = init_pts.cuda()

        pred_pts_list, pred_feats_list, pred_img = network(img, init_pts)

        my_img_loss = opt.lamb * total_img_loss(pred_img, img)
        my_pts_loss = total_pts_loss(pred_pts_list, pred_feats_list, pts, ellipsoid, use_cuda)

        loss = my_pts_loss + my_img_loss if epoch == 0 else my_pts_loss
        loss.backward()
        train_loss.update(loss.item())
        optimizer.step()

        if i % 50 == 0:
            vis.scatter(X = torch.squeeze(pts).data.cpu(),
                    win = 'TRAIN_INPUT',
                    opts = dict(
                        title = "TRAIN_INPUT",
                        markersize = 2,
                        ),
                    )
            vis.scatter(X = pred_pts_list[0].data.cpu(),
                    win = 'TRAIN_INPUT_RECONSTRUCTED_L1',
                    opts = dict(
                        title="TRAIN_INPUT_RECONSTRUCTED_L1",
                        markersize=2,
                        ),
                    )
            vis.scatter(X = pred_pts_list[1].data.cpu(),
                    win = 'TRAIN_INPUT_RECONSTRUCTED_L2',
                    opts = dict(
                        title="TRAIN_INPUT_RECONSTRUCTED_L2",
                        markersize=2,
                        ),
                    )
            vis.scatter(X = pred_pts_list[2].data.cpu(),
                    win = 'TRAIN_INPUT_RECONSTRUCTED_L3',
                    opts = dict(
                        title="TRAIN_INPUT_RECONSTRUCTED_L3",
                        markersize=2,
                        ),
                    )
            vis.image(img.data.cpu().squeeze(),
                    win = 'INPUT IMAGE',
                    opts = dict(
                        title = 'Input Image', 
                        caption = 'Input Image')
                    )
            vis.image(pred_img.data.cpu().squeeze(),
                    win = 'RECONSTRUCTED IMAGE',
                    opts = dict(
                        title = 'Reconstructed Image', 
                        caption = 'Reconstructed Image')
                    )
        
        print('[%d: %d/%d] train loss:  %f ' %(epoch, i, len_dataset, loss.item()))


    """

    # Validation
    val_loss.reset()
    network.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test, 0):
            img, pts, normal, _, _ = data
            init_pts = torch.from_numpy(ellipsoid['coord'])

            if use_cuda:
                img = img.cuda()
                pts = pts.cuda()
                normal = normal.cuda()
                init_pts = init_pts.cuda()

            pred_pts_list, pred_feats_list, pred_img = network(img, init_pts)

            my_img_loss = opt.lamb * total_img_loss(pred_img, img)
            my_pts_loss = total_pts_loss(pred_pts_list, pred_feats_list, pts, ellipsoid, use_cuda)

            loss = my_pts_loss + my_img_loss
            val_loss.update(loss.item())

            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
            
            if i % 200 ==0 :
                vis.scatter(X = torch.squeeze(pts).data.cpu(),
                        win = 'VAL_INPUT',
                        opts = dict(
                            title = "VAL_INPUT",
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = pred_pts_list[0].data.cpu(),
                        win = 'VAL_INPUT_RECONSTRUCTED_L1',
                        opts = dict(
                            title = "VAL_INPUT_RECONSTRUCTED_L1",
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = pred_pts_list[1].data.cpu(),
                        win = 'VAL_INPUT_RECONSTRUCTED_L2',
                        opts = dict(
                            title = "VAL_INPUT_RECONSTRUCTED_L2",
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = pred_pts_list[2].data.cpu(),
                        win = 'VAL_INPUT_RECONSTRUCTED_L3',
                        opts = dict(
                            title = "VAL_INPUT_RECONSTRUCTED_L3",
                            markersize = 2,
                            ),
                        )
                vis.image(img.data.cpu().squeeze(),
                    win = 'INPUT IMAGE',
                    opts = dict(
                        title = 'Input Image', 
                        caption = 'Input Image')
                    )
                vis.image(pred_img.data.cpu().squeeze(),
                    win = 'RECONSTRUCTED IMAGE',
                    opts = dict(
                        title = 'Reconstructed Image', 
                        caption = 'Reconstructed Image')
                    )

            print('[%d: %d/%d] val loss:  %f ' %(epoch, i, len(dataset_test), loss_net.item()))

    # Update visdom curve
    val_curve.append(val_loss.avg)

    vis.line(X = np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
             Y = np.column_stack((np.array(train_curve), np.array(val_curve))),
             win = 'loss',
             opts = dict(title = "loss", legend = ["train_curve", "val_curve"], markersize = 2, ), )
    vis.line(X = np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)))),
             Y = np.log(np.column_stack((np.array(train_curve), np.array(val_curve)))),
             win = 'log',
             opts = dict(title = "log", legend = ["train_curve", "val_curve"], markersize = 2, ), )

    #dump stats in log file
    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,
    }
    print(log_table)

    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    """
    #save last network
    print('saving net...')
    torch.save(network.state_dict(), '%s/network_%i.pth' % (dir_name, epoch))