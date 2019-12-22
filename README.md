# AlexNet-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained AlexNet models 
 * Use AlexNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an AlexNet on your own dataset
 * Export EfficientNet models for production
 
### Table of contents
1. [About AlexNet](#about-alexnet)
2. [Usage](#usage)
    * [Load models](#loading-models)
    * [train models](#train)
    * [eval models](#eval)
3. [Contributing](#contributing) 

### About AlexNet

If you're new to AlexNets, here is an explanation straight from the official PyTorch implementation: 

Current approaches to object recognition make essential use of machine learning methods. To improve their performance, we can collect larger datasets, learn more powerful models, and use better techniques for preventing overfitting. Until recently, datasets of labeled images were relatively
small — on the order of tens of thousands of images (e.g., NORB [16], Caltech-101/256 [8, 9], and
CIFAR-10/100 [12]). Simple recognition tasks can be solved quite well with datasets of this size,
especially if they are augmented with label-preserving transformations. For example, the currentbest error rate on the MNIST digit-recognition task (<0.3%) approaches human performance [4].
But objects in realistic settings exhibit considerable variability, so to learn to recognize them it is
necessary to use much larger training sets. And indeed, the shortcomings of small image datasets
have been widely recognized (e.g., Pinto et al. [21]), but it has only recently become possible to collect labeled datasets with millions of images. The new larger datasets include LabelMe [23], which
consists of hundreds of thousands of fully-segmented images, and ImageNet [6], which consists of
over 15 million labeled high-resolution images in over 22,000 categories. 

- **Resolution**

|     Datasets     |  Top1  |  Top5  | DataArgumentation |
|:----------------:|:------:|:------:|:-----------------:|
|CIFAR-10          | 71.24% | 97.05% |         √         |
|CIFAR-100         | 71.24% | 97.05% |         √         |

### Usage

#### Loading models

```text
if torch.cuda.device_count() > 1:
model = torch.nn.parallel.DataParallel(AlexNet(num_classes=opt.num_classes))
else:
model = AlexNet(num_classes=opt.num_classes)
if os.path.exists(MODEL_PATH):
model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
```

#### train

```text
python3 main --name <DATA_DIR> --num_classes <DATASETS_CLASSES> --phase train
```

#### eval

```text
python3 main --name <DATA_DIR> --num_classes <DATASETS_CLASSES>
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 