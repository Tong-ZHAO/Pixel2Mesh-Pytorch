import numpy as np
import torch
from gcn_layers import torch_sparse_tensor, dot

# Define Chamfer Loss
import sys
sys.path.append("model/chamfer/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()


def laplace_coord(input, lap_idx, block_id, use_cuda = True):
    
    # Inputs :
    # input : nodes Tensor, size (n_pts, n_features)
    # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
    # 
    # Returns : 
    # The laplacian coordinates of input with respect to edges as in lap_idx


    vertex = torch.cat((input, torch.zeros(1, 3).cuda()), 0) if use_cuda else torch.cat((input, torch.zeros(1, 3)), 0)
    
    indices = torch.tensor(lap_idx[block_id][:, :8])
    weights = torch.tensor(lap_idx[block_id][:,-1], dtype = torch.float32)

    if use_cuda:
        indices = indices.cuda()
        weights = weights.cuda()

    weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))

    num_pts, num_indices = indices.shape[0], indices.shape[1]
    indices = indices.reshape((-1,))
    vertices = torch.index_select(vertex, 0, indices)
    vertices = vertices.reshape((num_pts, num_indices, 3))

    laplace = torch.sum(vertices, 1)
    laplace = input - torch.mul(laplace, weights)
        
    return laplace

def laplace_loss(input1, input2, lap_idx, block_id, use_cuda = True):

    # Inputs : 
    # input1, input2 : nodes Tensor before and after the deformation
    # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
    # block_id : id of the deformation block (if different than 1 then adds
    # a move loss as in the original TF code)
    #
    # Returns :
    # The Laplacian loss, with the move loss as an additional term when relevant

    lap1 = laplace_coord(input1, lap_idx, block_id, use_cuda)
    lap2 = laplace_coord(input2, lap_idx, block_id, use_cuda)
    laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 1500
    move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2), 1)) * 100

    if block_id == 0:
        return laplace_loss
    else:
        return laplace_loss + move_loss



def edge_loss(pred, gt_pts, edges, block_id, use_cuda = True):

	# edge in graph
    #nod1 = pred[edges[block_id][:, 0]]
    #nod2 = pred[edges[block_id][:, 1]]
    idx1 = torch.tensor(edges[block_id][:, 0]).long()
    idx2 = torch.tensor(edges[block_id][:, 1]).long()

    if use_cuda:
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

    nod1 = torch.index_select(pred, 0, idx1)
    nod2 = torch.index_select(pred, 0, idx2)
    edge = nod1 - nod2

	# edge length loss
    edge_length = torch.sum(torch.pow(edge, 2), 1)
    edge_loss = torch.mean(edge_length) * 300

    return edge_loss


def L1Tensor(img1, img2) : 
	""" input shoudl be tensor and between 0 and 1"""
	mae = torch.mean(torch.abs(img2 - img1))
	return mae


def L2Tensor(img1, img2) : 
	""" input shoudl be tensor and between 0 and 1"""
	mse = torch.mean((img2 - img1) ** 2)
	return mse


def total_pts_loss(pred_pts_list, pred_feats_list, gt_pts, ellipsoid, use_cuda = True):
    """
    pred_pts_list: [x1, x1_2, x2, x2_2, x3]
    """

    my_chamfer_loss, my_edge_loss, my_lap_loss = 0., 0., 0.
    lap_const = [0.2, 1., 1.]

    for i in range(3):
        dist1, dist2 = distChamfer(gt_pts, pred_pts_list[i].unsqueeze(0))
        my_chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
        my_edge_loss += edge_loss(pred_pts_list[i], gt_pts, ellipsoid["edges"], i, use_cuda)
        my_lap_loss += lap_const[i] * laplace_loss(pred_feats_list[i], pred_pts_list[i], ellipsoid["lap_idx"], i, use_cuda)

    my_pts_loss = 100 * my_chamfer_loss + 0.1 * my_edge_loss + 0.3 * my_lap_loss

    return my_pts_loss



def total_img_loss(pred_img, gt_img):

    my_rect_loss = torch.nn.functional.binary_cross_entropy(pred_img, gt_img, size_average = False)
    my_l1_loss = L1Tensor(pred_img, gt_img)

    img_loss = my_rect_loss + my_l1_loss

    return img_loss
    
    



if False:#__name__ == '__main__':
    
    # Test laplacian losses
    
    input1 = torch.rand((100,4,3))
    input2 = torch.rand((100,4,3))
    adj = torch.randint(2, (100,4,4))
    for ind1 in range(4):
        for ind2 in range(4):
            if ind1 == ind2:
                adj[:,ind1,ind2] = 0
            else:
                adj[:,ind1,ind2] = torch.max(adj[:,ind1,ind2], adj[:,ind2,ind1])
    
    print(input1, input2, adj)
    
    loss = laplace_loss(input1, input2, adj, 2)
    print(loss)