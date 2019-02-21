# Pixel2Mesh-Pytorch

This repository aims to implement the ECCV 2018 paper: [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](http://bigvid.fudan.edu.cn/pixel2mesh/) in PyTorch. The [official code](https://github.com/nywang16/Pixel2Mesh) in Tensorflow is available online. Based on the proposed structure, we replaced the VGG model by a U-Net based autoencoder to reconstruct the image, which helps the net to converge faster.

<img src="/img/net.png" width="900"/>

## Requirements

* PyTorch 1.0 (Enable Sparse Tensor)
* \>= Python 3
* \>= Cuda 9.2 (Enable Chamfer Distance)
* Visdom (Enable Data Visualization)

## External Codes

* [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn): Base code of GraphConvolution
* [atlasnet](https://github.com/ThibaultGROUEIX/AtlasNet/tree/master/extension): Chamfer Distance

## Getting Started

### Installation

```
cd ./model/chamfer/
python setup.py install
```

### Dataset

We use the same dataset as the one used in Pixel2Mesh. The point clouds are from [ShapeNet](https://www.shapenet.org/) and the rendered views are from [3D-R2N2](https://github.com/chrischoy/3D-R2N2).

The whole dataset can be downloaded [Here](https://drive.google.com/file/d/1Z8gt4HdPujBNFABYrthhau9VZW10WWYe/view?usp=sharing).

Please respect the [ShapeNet license](https://shapenet.org/terms) while using.

### Train

```
python train.py
```

The hyper-parameters can be changed from command. To get more help, please use

```
python train.py -h
```

### Validation

To evaluate the model on one example, please use the following command

```
python evaluate.py --dataPath *** --modelPath ***
```

## Some Examples

Due to the device limit, we trained our model on the airplane class instead of the whole dataset. A trained model is provided [Here](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch/blob/master/log/2019-01-18T02:32:24.320546/network_4.pth)

Some test examples are shown as below:

<img src="/img/examples_1.png" width="900"/>

<img src="/img/examples_2.png" width="900"/>


