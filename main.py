# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import random

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.dataloader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url

from model import AlexNet
from utils.eval import accuracy
from utils.misc import AverageMeter

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

parser = argparse.ArgumentParser(description='PyTorch AlexNet Classifier')
parser.add_argument('--dataroot', type=str,
                    default="./datasets/", help="dataset path.")
parser.add_argument('--dataname', type=str, default="fruits",
                    help="Dataset name. Default: fruits.")
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int,
                    default=128, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=224,
                    help='the height / width of the inputs image to network')
parser.add_argument('--lr', type=float, default=0.0001,
                    help="learning rate.")
parser.add_argument('--epochs', type=int, default=50, help="Train loop")
parser.add_argument('--phase', type=str, default='eval',
                    help="train or eval? default:`eval`")
parser.add_argument('--checkpoints_dir', default='./checkpoints',
                    help='folder to output model checkpoints')
parser.add_argument('--pretrained', type=bool, default=False,
                    help='If `True`, load pre trained model weights. Default: `False`.')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

try:
    os.makedirs(opt.checkpoints_dir)
except OSError:
    pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train datasets path
TRAIN_DATASETS_PATH = os.path.join(opt.dataroot, f"{opt.dataname}/train")
# test datasets path
TEST_DATASETS_PATH = os.path.join(opt.dataroot, f"{opt.dataname}/test")

# model path
MODEL_PATH = os.path.join(opt.checkpoints_dir, f"{opt.dataname}.pth")

train_dataset = dset.ImageFolder(root=TRAIN_DATASETS_PATH,
                                 transform=transforms.Compose([
                                     transforms.Resize(
                                         (opt.img_size, opt.img_size), interpolation=3),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [
                                                          0.229, 0.224, 0.225]),
                                 ]))
test_dataset = dset.ImageFolder(root=TEST_DATASETS_PATH,
                                transform=transforms.Compose([
                                    transforms.Resize(
                                        (opt.img_size, opt.img_size), interpolation=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [
                                                         0.229, 0.224, 0.225]),
                                ]))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=int(opt.workers))


def train():
    try:
        os.makedirs(opt.checkpoints_dir)
    except OSError:
        pass
    if torch.cuda.device_count() > 1:
      model = torch.nn.parallel.DataParallel(AlexNet(**kwargs))
    else:
      model = AlexNet(num_classes=opt.num_classes)
    if pretrained:
      state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                            progress=progress)
      model.load_state_dict(state_dict)

    ################################################
    # Set loss function and Adam optimizer
    ################################################
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        # train for one epoch
        print(f"\nBegin Training Epoch {epoch + 1}")
        # Calculate and return the top-k accuracy of the model
        # so that we can track the learning process.
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # compute gradients in a backward pass
            optimizer.zero_grad()
            loss.backward()

            # Call step of optimizer to update model params
            optimizer.step()

            print(f"Epoch [{epoch + 1}] [{i + 1}/{len(train_dataloader)}]\t"
                  f"Loss {loss.item():.4f}\t"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                  f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})", end="\r")

        # save model file
        torch.save(CNN.state_dict(), MODEL_PATH)


def test():
    CNN = AlexNet(num_classes=120)
    CNN.load_state_dict(torch.load(MODEL_PATH))
    CNN.to(device)
    CNN.eval()

    # init value
    total = 0.
    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = CNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100 * correct / total
    return acc


if __name__ == '__main__':
    if opt.dataname is None:
        print(parser.print_help())
        exit(0)
    elif opt.phase == "train":
        train()
    elif opt.phase == "eval":
        print("Loading model successful!")
        accuracy = test()
        print(
            f"\nAccuracy of the network on the test images: {accuracy:.2f}%.\n")
