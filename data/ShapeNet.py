import numpy as np
import pickle
import os, sys

from PIL import Image

import torchvision
import torch.utils.data as data
from torchvision.transforms import Resize

word_idx = {'02691156': 0, # airplane
            '03636649': 1, # lamp
            '03001627': 2} # chair

idx_class = {0: 'airplane', 1: 'lamp', 2: 'chair'}


class ShapeNet(data.Dataset):
    """Dataset wrapping images and target meshes for ShapeNet dataset.

    Arguments:
    """

    def __init__(self, file_root, file_list):

        self.file_root = file_root
        # Read file list
        with open(file_list, "r") as fp:
            self.file_names = fp.read().split("\n")[:-1]
        self.file_nums = len(self.file_names)


    def __getitem__(self, index):

        name = os.path.join(self.file_root, self.file_names[index])
        data = pickle.load(open(name, "rb"), encoding = 'latin1')
        img, pts, normals = data[0].astype('float32') / 255.0, data[1][:, :3], data[1][:, 3:]
        img = np.transpose(img, (2, 0, 1))
        label = word_idx[self.file_names[index].split('_')[0]]

        return img, pts, normals, label, self.file_names[index]


    def __len__(self):
        return self.file_nums



if __name__ == "__main__":

    file_root = "/home/pkq/tong/MVA/S1/objet/project/data/ShapeNetSmall"
    dataloader = ShapeNet(file_root, 'train_list_small.txt')

    print("Load %d files!\n" % len(dataloader))

    img, pts, normals, label, name = dataloader[0]

    print("Info for the first data:")
    print("Image Shape: ", img.shape)
    print("Point cloud shape: ", pts.shape)
    print("Normal shape: ", normals.shape)
    print("Class: ", idx_class[label])
    print("File name: ", name)