## Deep Convolution Generative Adversarial Networks

### Introduction

This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

The implementation is very close to the Torch implementation [main.py](https://github.com/Lornatang/PyTorch-DCGAN/main.py)

After every 100 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

#### Configure

- [PyTorch](https://pytorch.org) > 1.3.0
- GTX 1080 Ti

### Load dataset

- [baidu netdisk](https://pan.baidu.com/s/1eSifHcA) password：`g5qa`

**download data put on ./datasets folder.**

Thanks [何之源](https://www.zhihu.com/people/he-zhi-yuan-16)

```text
datasets/
└── faces/
    ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
    ├── 0001a0fca4e9d2193afea712421693be-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-1.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-0.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-1.jpg
    ├── 000333906d04217408bb0d501f298448-0.jpg
    ├── 0005027ac1dcc32835a37be806f226cb-0.jpg
```

#### Purpose

Use a stable DCGAN structure to generate avatar images of anime girls.

#### Usage

- train

if you want pretrain generate model, 
click it **[netg_200.pth](http://pytorch-1252820389.cosbj.myqcloud.com/netg_200.pth)**

if you want pretrain discriminate model, 
click it **[netd_200.pth](http://pytorch-1252820389.cosbj.myqcloud.com/netd_200.pth)**

please rename model name. `netd_200.pth` -> `D.pth` and `netg_200.pth` -> `G.pth`

start run:
```text
python main.py
```

- test

```text
python main.py --phase generate
```

#### Example

- epoch 1

![epoch1.png](https://github.com/Lornatang/PyTorch-DCGAN/blob/master/assets/epoch1.png)

- epoch 30

![epoch30.png](https://github.com/Lornatang/PyTorch-DCGAN/blob/master/assets/epoch30.png)

- epoch 100

![epoch100.png](https://github.com/Lornatang/PyTorch-DCGAN/blob/master/assets/epoch100.png)

- epoch 200

![epoch200.png](https://github.com/Lornatang/PyTorch-DCGAN/blob/master/assets/epoch200.png)
