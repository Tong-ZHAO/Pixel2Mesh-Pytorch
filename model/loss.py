import numpy as np
import torch
from gcn_layers import torch_sparse_tensor, dot


def laplace_coord(input, adj):
    
    # Inputs :
    # input : nodes Tensor, size (n_batch, n_pts, n_features)
    # adj : zero-one edges matrix Tensor, size (n_batch, n_pts, n_pts)
    # 
    # Returns : 
    # The laplacian coordinates of input with respect to edges as in adj
    
    adj_sum = torch.sum(adj, 2)
    adj_sum = adj_sum.view(adj_sum.shape[0], adj_sum.shape[1], 1)
    adj_new = torch.div(adj, adj_sum)
    
    lap = input - dot(adj_new, input, True)
        
    return lap

def laplace_loss(input1, input2, adjs, block_id, use_cuda = True):
    
    # Inputs : 
    # input1, input2 : nodes Tensor before and after the deformation
    # adjs : (adjs[0], adjs[1]) where adjs[1]: non-negative edges matrix Tensor, size (n_batch, n_pts, n_pts)
    # block_id : id of the deformation block (if different than 1 then adds
    # a move loss as in the original TF code)
    #
    # Returns :
    # The Laplacian loss, with the move loss as an additional term when relevant
    
    adj = (torch_sparse_tensor(*adjs[1], use_cuda) > 0)
    
    lap1 = laplace_coord(input1, adj)
    lap2 = laplace_coord(input2, adj)
    
    laplace_loss = torch.mean(torch.sum( torch.pow(lap1-lap2,2), (1,2))) * 1500
    
    if block_id == 1:
        move_loss = torch.Tensor(0.)
    else:
        move_loss = torch.mean(torch.sum( torch.pow(input1-input2,2), (1,2))) * 100
    
    return laplace_loss + move_loss

def edge_length_loss(input, adjs, use_cuda = True):
    
    # Inputs :
    # input : nodes Tensor, size (n_batch, n_pts, n_features)
    # adjs : (adjs[0], adjs[1]) where adjs[1]: non-negative edges matrix Tensor, size (n_batch, n_pts, n_pts)
    
    adj = (torch_sparse_tensor(*adjs[1], use_cuda) > 0).repeat((1,1,1,3))
    input1 = input.repeat((1,1,input.shape[1],1))
    input2 = input.repeat((1,input.shape[1],1,1))
    diff = torch.mul(adj,input1-input2)
    
    loss = torch.mean(torch.sum( torch.pow(diff,2), (1,2,3))) * 300
    
    return loss


def L1Tensor(img1, img2) : 
	""" input shoudl be tensor and between 0 and 1"""
	mae = torch.mean(torch.abs(img2 - img1))
	return mae


def L2Tensor(img1, img2) : 
	""" input shoudl be tensor and between 0 and 1"""
	mse = torch.mean((img2 - img1) ** 2)
	return mse
    
    

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