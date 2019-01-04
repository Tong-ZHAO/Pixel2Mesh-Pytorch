import numpy as np
import pickle


def read_init_mesh(file):

    with open(file, "rb") as fp:
        fp_info = pickle.load(fp, encoding = 'latin1')

    # shape: n_pts * 3
    init_coord = fp_info[0]

    # edges & faces & lap_idx
    # edge: num_edges * 2
    # faces: num_faces * 4
    # lap_idx: num_pts * 10
    edges, faces, lap_idx = [], [], []

    for i in range(3):
        edges.append(fp_info[1 + i][1][0])
        faces.append(fp_info[5][i])
        lap_idx.append(fp_info[7][i])

    # pool index
    # num_pool_edges * 2
    pool_idx = [fp_info[4][0], fp_info[4][1]] # pool_01: 462 * 2, pool_12: 1848 * 2

    # supports
    # 0: np.array, 2D, pos
    # 1: np.array, 1D, vals
    # 2: tuple - shape, n * n
    support1, support2, support3 = [], [], []

    for i in range(2):
        support1.append(fp_info[1][i])
        support2.append(fp_info[2][i])
        support3.append(fp_info[3][i])
        
    keys = ["coord", "edges", "faces", "lap_idx", "pool_idx", "supports"]
    vals = [init_coord, edges, faces, lap_idx, pool_idx, [support1, support2, support3]]

    return dict(zip(keys, vals))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('GraphConv') != -1:
        m.reset_parameters()
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    