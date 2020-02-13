# AlexNet-PyTorch

### Update (Feb 13, 2020)

The update is for easy expansion

 * [Example: Export to ONNX](#example-export-to-onnx)
 * [Example: Extract features](#example-feature-extraction)

It is also now incredibly simple to load a pretrained model with a new number of classes for transfer learning:
```python
from alexnet_pytorch import AlexNet
model = AlexNet.from_pretrained('alexnet', num_classes=10)
``` 

### Update (January 15, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

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
2. [Model Description](#model-description)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export-to-onnx)
5. [Contributing](#contributing) 

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

### Model Description

AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) during training.

The 1-crop error rates on the imagenet dataset with the pretrained model are listed below.

|Model structure|*Top-1 error*|*Top-5 error*|
|:-------------:|:-----------:|:-----------:|
|    alexnet    |    43.48    |    20.93    |

### Installation

Install from source:
```bash
git clone https://github.com/lornatang/AlexNet-PyTorch
cd AlexNet-PyTorch
pip install -e .
``` 

### Usage

#### Loading pretrained models

Load an AlexNet:  
```python
from alexnet_pytorch import AlexNet
model = AlexNet.from_name('alexnet')
```

Load a pretrained AlexNet: 
```python
from alexnet_pytorch import AlexNet
model = AlexNet.from_pretrained('alexnet')
```

#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from alexnet_pytorch import AlexNet

# Open image
input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with AlexNet
model = AlexNet.from_pretrained("alexnet")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
  input_batch = input_batch.to("cuda")
  model.to("cuda")

with torch.no_grad():
  logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
  label = labels_map[idx]
  prob = torch.softmax(logits, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
```

#### Example: Feature Extraction 

You can easily extract features with `model.extract_features`:
```python
import torch
from alexnet_pytorch import AlexNet
model = AlexNet.from_pretrained('alexnet')

# ... image preprocessing as in the classification example ...
inputs = torch.randn(1, 3, 224, 224)
print(inputs.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(inputs)
print(features.shape) # torch.Size([1, 256, 6, 6])
```

#### Example: Export to ONNX  

Exporting to ONNX for deploying to production is now simple: 
```python
import torch 
from alexnet_pytorch import AlexNet

model = AlexNet.from_pretrained('alexnet')
dummy_input = torch.randn(16, 3, 224, 224)

torch.onnx.export(model, dummy_input, "demo.onnx", verbose=True)
``` 

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

For more datasets result. Please see `research/README.md`.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 
