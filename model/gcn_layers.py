import math
import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from scipy.sparse import coo_matrix


def torch_sparse_tensor(indice, value, size, use_cuda):

    coo = coo_matrix((value, (indice[:, 0], indice[:, 1])), shape = size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if use_cuda:
        return torch.sparse.FloatTensor(i, v, shape).cuda()
    else:
        return torch.sparse.FloatTensor(i, v, shape)


def dot(x, y, sparse = False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        res = x.mm(y)
    else:
        res = torch.matmul(x, y)
    return res


class GraphConvolution(Module):
    """Simple GCN layer
    
    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adjs, bias=True, use_cuda = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        adj0 = torch_sparse_tensor(*adjs[0], use_cuda)
        adj1 = torch_sparse_tensor(*adjs[1], use_cuda)
        self.adjs = [adj0, adj1]

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        #output = torch.spmm(adj, support)
        output1 = dot(self.adjs[0], support, True)
        output2 = dot(self.adjs[1], support, True)
        output = output1 + output2
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphPooling(Module):
    """Graph Pooling layer, aims to add additional vertices to the graph.

    The middle point of each edges are added, and its feature is simply
    the average of the two edge vertices.
    Three middle points are connected in each triangle.
    """

    def __init__(self, pool_idx):
        super(GraphPooling, self).__init__() 
        self.pool_idx = pool_idx
        # save dim info
        self.in_num = np.max(pool_idx)
        self.out_num = self.in_num + len(pool_idx)

    def forward(self, input):

        new_features = input[self.pool_idx].clone()
        new_vertices = 0.5 * new_features.sum(1)
        output = torch.cat((input, new_vertices), 0)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_num) + ' -> ' \
               + str(self.out_num) + ')'



class GraphProjection(Module):
    """Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use 
    bilinear interpolation to get the corresponding feature.
    """

    def __init__(self):
        super(GraphProjection, self).__init__()


    def forward(self, img_features, input):

        self.img_feats = img_features 

        h = 248 * torch.div(input[:, 1], input[:, 2]) + 111.5
        w = 248 * torch.div(input[:, 0], -input[:, 2]) + 111.5

        h = torch.clamp(h, min = 0, max = 223)
        w = torch.clamp(w, min = 0, max = 223)

        img_sizes = [56, 28, 14, 7]
        out_dims = [64, 128, 256, 512]
        feats = [input]

        for i in range(4):
            out = self.project(i, h, w, img_sizes[i], out_dims[i])
            feats.append(out)
            
        output = torch.cat(feats, 1)
        
        return output

    def project(self, index, h, w, img_size, out_dim):

        img_feat = self.img_feats[index]
        x = h / (224. / img_size)
        y = w / (224. / img_size)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max = img_size - 1)
        y2 = torch.clamp(y2, max = img_size - 1)

        #Q11 = torch.index_select(torch.index_select(img_feat, 1, x1), 1, y1)
        #Q12 = torch.index_select(torch.index_select(img_feat, 1, x1), 1, y2)
        #Q21 = torch.index_select(torch.index_select(img_feat, 1, x2), 1, y1)
        #Q22 = torch.index_select(torch.index_select(img_feat, 1, x2), 1, y2)

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q12, 0 ,1))

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        return output
