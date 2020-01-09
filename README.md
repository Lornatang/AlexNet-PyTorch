# AlexNet-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained AlexNet models 
 * Use AlexNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an AlexNet on your own dataset
 * Export AlexNet models for production
 
### Table of contents
1. [About AlexNet](#about-alexnet)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
4. [Contributing](#contributing) 

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

### Installation

Install from source:
```bash
git clone https://github.com/lornatang/AlexNet-PyTorch
cd AlexNet-Pytorch
pip install -e .
``` 

### Usage

#### Loading pretrained models

Load an AlexNet:  
```python
from alexnet import AlexNet
model = AlexNet()
```

Load a pretrained AlexNet: 
```python
from alexnet import AlexNet
model = AlexNet.from_pretrained('alexnet')
```

Details about the models are below: 

|   *DatasetName*   |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
|     `cifar10`     |    57M   |    79.1    |      ✓      |
|     `cifar100`    |   57.4M  |    47.4    |      ✓      |


#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

```python
import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from alexnet import AlexNet

image_size = 224

# Open image
img = Image.open('panda.jpg')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
img = tfms(img).unsqueeze(0)

# Load class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with AlexNet
print("=> loading checkpoint 'alexnet'.")
model = AlexNet.from_pretrained('alexnet-e3')
print("=> loaded checkpoint 'alexnet'.")
model.eval()
with torch.no_grad():
    logits = model(img)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print('-----')
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(label, prob * 100))
```

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 