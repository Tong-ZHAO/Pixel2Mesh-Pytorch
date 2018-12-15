import numpy as np
import pickle
import os, sys

from PIL import Image

import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize


class ShapeNet(data.Dataset):
    """Dataset wrapping images and target meshes for ShapeNet dataset.

    Arguments:
    """

    def __init__(self, file_root, file_list):

        self.file_root = file_root
        # Read file list
        with open(file_list, "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.file_nums = len(file_names)
        np.random.shuffle(self.file_names)

    def __getitem__(self, index):

        name = os.path.join(self.file_root, self.file_names[index])
        data = pickle.load(open(name, "rb"), encoding = 'latin1')
        img, pts, normals = data[0].astype('float32') / 255.0, data[1][:, :3], data[1][:, 3:]

        return img, pts, normals, self.file_names[index]


    def __len__(self):
        return len(self.file_nums)



if __name__ == "__main__":

    file_root = "/home/pkq/tong/MVA/S1/objet/project/data/ShapeNetTrain"
    