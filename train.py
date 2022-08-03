<<<<<<< Updated upstream
# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import os
import shutil
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import CUDAPrefetcher, ImageDataset
from model import AlexNet
from torch.optim.swa_utils import AveragedModel
from torch.optim import lr_scheduler
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    model, ema_model = build_model()
    print("Build AlexNet model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_path:
        model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(model,
                                                                              ema_model,
                                                                              config.pretrained_model_path,
                                                                              "pretrained",
                                                                              start_epoch,
                                                                              best_acc1,
                                                                              optimizer,
                                                                              scheduler)
        print(f"Loaded `{config.pretrained_model_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(model,
                                                                              ema_model,
                                                                              config.pretrained_model_path,
                                                                              "resume",
                                                                              start_epoch,
                                                                              best_acc1,
                                                                              optimizer,
                                                                              scheduler)
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer)
        acc1 = validate(model, valid_prefetcher, epoch, writer, "Valid")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        torch.save({"epoch": epoch + 1,
                    "best_acc1": best_acc1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir, config.image_size, "Train")
    valid_dataset = ImageDataset(config.valid_image_dir, config.image_size, "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    model = AlexNet(config.model_num_classes)
    model = model.to(device=config.device, memory_format=torch.channels_last)

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
                                                                                          1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_model = AveragedModel(model, avg_fn=ema_avg)

    return model, ema_model


def define_loss() -> nn.CrossEntropyLoss:
    pixel_criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    pixel_criterion = pixel_criterion.to(device=config.device, memory_format=torch.channels_last)

    return pixel_criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                         config.lr_scheduler_T_0,
                                                         config.lr_scheduler_T_mult,
                                                         config.lr_scheduler_eta_min)

    return scheduler


def train(model: nn.Module,
          train_prefetcher: CUDAPrefetcher,
          criterion: nn.CrossEntropyLoss,
          optimizer: optim.Adam,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        acc1.update(top1[0], batch_size)
        acc5.update(top5[0], batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Mixed precision training
            with amp.autocast():
                output = model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0], batch_size)
            acc5.update(top5[0], batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1.avg


if __name__ == "__main__":
    main()
=======
# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import os
import shutil
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import CUDAPrefetcher, ImageDataset
from model import AlexNet
from torch.optim.swa_utils import AveragedModel
from torch.optim import lr_scheduler
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_acc1 = 0.0

    train_prefetcher, valid_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    model, ema_model = build_model()
    print("Build AlexNet model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_path:
        model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(model,
                                                                              ema_model,
                                                                              config.pretrained_model_path,
                                                                              "pretrained",
                                                                              start_epoch,
                                                                              best_acc1,
                                                                              optimizer,
                                                                              scheduler)
        print(f"Loaded `{config.pretrained_model_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(model,
                                                                              ema_model,
                                                                              config.pretrained_model_path,
                                                                              "resume",
                                                                              start_epoch,
                                                                              best_acc1,
                                                                              optimizer,
                                                                              scheduler)
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(model, ema_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer)
        acc1 = validate(ema_model, valid_prefetcher, epoch, writer, "Valid")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        torch.save({"epoch": epoch + 1,
                    "best_acc1": best_acc1,
                    "state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir, config.image_size, "Train")
    valid_dataset = ImageDataset(config.valid_image_dir, config.image_size, "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    model = AlexNet(config.model_num_classes)
    model = model.to(device=config.device, memory_format=torch.channels_last)

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
                                                                                          1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_model = AveragedModel(model, avg_fn=ema_avg)

    return model, ema_model


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)

    return criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                         config.lr_scheduler_T_0,
                                                         config.lr_scheduler_T_mult,
                                                         config.lr_scheduler_eta_min)

    return scheduler


def train(
        model: nn.Module,
        ema_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_model.update_parameters(model)

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        acc1.update(top1[0], batch_size)
        acc5.update(top5[0], batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        ema_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"{mode}: ")

    # Put the exponential moving average model in the verification mode
    ema_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Mixed precision training
            with amp.autocast():
                output = ema_model(images)

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0], batch_size)
            acc5.update(top5[0], batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc1.avg


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
