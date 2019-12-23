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

from model import AlexNet
from utils.eval import accuracy
from utils.misc import AverageMeter
from utils.adjust import adjust_learning_rate

parser = argparse.ArgumentParser(description='PyTorch AlexNet Classifier')
parser.add_argument('--dataroot', type=str,
                    default="./datasets/", help="dataset path.")
parser.add_argument('--name', type=str, default="fruits",
                    help="Dataset name. Default: fruits.")
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=1)
parser.add_argument('--batch_size', type=int,
                    default=128, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=224,
                    help='the height / width of the inputs image to network')
parser.add_argument('--num_classes', type=int, default=1000,
                    help="number of dataset category.")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate. Default: 0.001. For use adjust learning rate.")
parser.add_argument('--epochs', type=int, default=2000, help="Train loop")
parser.add_argument('--phase', type=str, default='eval',
                    help="train or eval? default:`eval`")
parser.add_argument('--checkpoints_dir', default='./checkpoints',
                    help='folder to output model checkpoints')
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
TRAIN_DATASETS_PATH = os.path.join(opt.dataroot, f"{opt.name}/train")
# test datasets path
TEST_DATASETS_PATH = os.path.join(opt.dataroot, f"{opt.name}/test")

# model path
MODEL_PATH = os.path.join(opt.checkpoints_dir, f"{opt.name}.pth")


def train():
  try:
    os.makedirs(opt.checkpoints_dir)
  except OSError:
    pass

  train_dataset = dset.ImageFolder(root=TRAIN_DATASETS_PATH,
                                    transform=transforms.Compose([
                                    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                                    transforms.Resize((opt.img_size, opt.img_size), interpolation=3),
                                    transforms.RandomRotation(degrees=15),
                                    transforms.ColorJitter(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                  ]))
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                 pin_memory=torch.cuda.is_available(),
                                                 shuffle=True, num_workers=int(opt.workers))

  if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DataParallel(AlexNet(num_classes=opt.num_classes))
  else:
    model = AlexNet(num_classes=opt.num_classes)
  if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
  model.to(device)
  ################################################
  # Set loss function and Adam optimizer
  ################################################
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

  for epoch in range(opt.epochs):
    # train for one epoch
    print(f"\nBegin Training Epoch {epoch + 1}")
    # Sets the learning rate to the initial LR decayed by 0.97 every 2.4 epochs
    adjust_learning_rate(opt.lr, optimizer, epoch)
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
    torch.save(model.state_dict(), MODEL_PATH)


def test():
  test_dataset = dset.ImageFolder(root=TEST_DATASETS_PATH,
                                  transform=transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size), interpolation=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                  ]))
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                pin_memory=torch.cuda.is_available(),
                                                num_workers=int(opt.workers))

  if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DataParallel(AlexNet(num_classes=opt.num_classes))
  else:
    model = AlexNet(num_classes=opt.num_classes)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
  model.to(device)

  # init value
  correct1 = 0.
  correct5 = 0.
  total = len(test_dataloader.dataset)
  with torch.no_grad():
    for i, data in enumerate(test_dataloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, targets = data
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)

      # cal top 1 accuracy
      prec1 = outputs.argmax(dim=1)
      correct1 += torch.eq(prec1, targets).sum().item()

      # cal top 5 accuracy
      maxk = max((1, 5))
      targets_resize = targets.view(-1, 1)
      _, prec5 = outputs.topk(maxk, 1, True, True)
      correct5 += torch.eq(prec5, targets_resize).sum().item()

  return correct1 / total, correct5 / total


if __name__ == '__main__':
  if opt.name is None:
    print(parser.print_help())
    exit(0)
  elif opt.phase == "train":
    train()
  elif opt.phase == "eval":
    print("Loading model successful!")
    acc1, acc5 = test()
    print(
      f"Top 1 accuracy of the network on the test images: {acc1:.6f}.\n"
      f"Top 5 accuracy of the network on the test images: {acc5:.6f}.\n")
