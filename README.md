# AlexNet-PyTorch

<a href="https://console.tiyaro.ai/explore/trn:model:123456789012-venkat:1.0:alexnet_pytorch_6c50c5">
<img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/tiyaro_badge.svg"></a>


## Overview

This repository contains an op-for-op PyTorch reimplementation of [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

## Table of contents

- [AlexNet-PyTorch](#alexnet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [ImageNet Classification with Deep Convolutional Neural Networks](#imagenet-classification-with-deep-convolutional-neural-networks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file. 

### Test

- line 29: `model_num_classes` change to `1000`.
- line 31: `mode` change to `test`.
- line 79: `model_path` change to `./results/pretrained_models/AlexNet-ImageNet_1K-9df8cd0f.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_num_classes` change to `1000`.
- line 31: `mode` change to `train`.
- line 33: `exp_name` change to `AlexNet-ImageNet_1K`.
- line 45: `pretrained_model_path` change to `./results/pretrained_models/AlexNet-ImageNet_1K-9df8cd0f.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_num_classes` change to `1000`.
- line 31: `mode` change to `train`.
- line 33: `exp_name` change to `AlexNet-ImageNet_1K`.
- line 48: `resume` change to `./samples/AlexNet-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|  Model  |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:-------:|:-----------:|:-----------------:|:-----------------:|
| AlexNet | ImageNet_1K | 36.7%(**43.8%**)  | 15.4%(**21.3%**)  |

```bash
# Download `AlexNet-ImageNet_1K-9df8cd0f.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input: 

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output: 

```text
Build AlexNet model successfully.
Load AlexNet model weights `/AlexNet-PyTorch/results/pretrained_models/AlexNet-ImageNet_1K-9df8cd0f.pth.tar` successfully.
tench, Tinca tinca                                                          (95.73%)
bolete                                                                      (1.20%)
triceratops                                                                 (0.43%)
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus (0.36%)
croquet ball                                                                (0.28%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### ImageNet Classification with Deep Convolutional Neural Networks

*Alex Krizhevsky,Ilya Sutskever,Geoffrey E. Hinton*

##### Abstract

We trained a large, deep convolutional neural network to classify the 1.2 million
high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5%
and 17.0% which is considerably better than the previous state-of-the-art. The
neural network, which has 60 million parameters and 650,000 neurons, consists
of five convolutional layers, some of which are followed by max-pooling layers,
and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected
layers we employed a recently-developed regularization method called “dropout”
that proved to be very effective. We also entered a variant of this model in the
ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%,
compared to 26.2% achieved by the second-best entry.

[[Paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

```bibtex
@article{AlexNet,
    title = {ImageNet Classification with Deep Convolutional Neural Networks},
    author = {Alex Krizhevsky,Ilya Sutskever,Geoffrey E. Hinton},
    journal = {nips},
    year = {2012}
}
```
