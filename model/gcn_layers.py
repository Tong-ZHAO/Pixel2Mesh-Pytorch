import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """Simple GCN layer
    
    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
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

        new_vertices = 0.5 * torch.gather(input, index = self.pool_idx).sum(1)
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
        w = 248 * tf.divide(input[:, 0], -input[:, 2]) + 111.5

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

        x1, x2 = torch.floor(x).int(), torch.ceil(x).int()
        y1, y2 = torch.floor(y).int(), torch.ceil(y).int()

        Q11 = torch.index_select(torch.index_select(img_feat, 0, x1), 0, y1)
        Q12 = torch.index_select(torch.index_select(img_feat, 0, x1), 0, y2)
        Q21 = torch.index_select(torch.index_select(img_feat, 0, x2), 0, y1)
        Q22 = torch.index_select(torch.index_select(img_feat, 0, x2), 0, y2)

        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.view(-1, 1), Q11)

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.view(-1, 1), Q12)

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.view(-1, 1), Q21)

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.view(-1, 1), Q22)

        output = Q11 + Q21 + Q12 + Q22

        return outputs
