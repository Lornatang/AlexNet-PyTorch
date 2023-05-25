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
import time

import torch
from torch import nn
from torch import optim ## Import optim module from torch library
from torch.cuda import amp ## Import amp module from torch library 
from torch.optim import lr_scheduler ## Import lr_scheduler module from torch library
from torch.optim.swa_utils import AveragedModel ## Import AveragedModel module from torch library
from torch.utils.data import DataLoader ## Import DataLoader module from torch library
from torch.utils.tensorboard import SummaryWriter ## Import SummaryWriter module from torch library

import config
import model
from dataset import CUDAPrefetcher, ImageDataset ## Import the modules from dataset.py file
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0 ## The training should start from epoch 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0 ## Keeps count of the current best accuracy, initialized with base value of 0

    train_prefetcher, valid_prefetcher = load_dataset()
    print("Load all datasets successfully.") 

    alexnet_model, ema_alexnet_model = build_model()
    print(f"Build {config.model_arch_name.upper()} model successfully.")

    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(alexnet_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        alexnet_model, ema_alexnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(alexnet_model,
                                                                                                         config.pretrained_model_weights_path,
                                                                                                         ema_alexnet_model,
                                                                                                         start_epoch,
                                                                                                         best_acc1,
                                                                                                         optimizer,
                                                                                                         scheduler)
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume: ## undefined
        alexnet_model, ema_alexnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(alexnet_model, ## undefined
                                                                                                         config.pretrained_model_weights_path, ## undefined
                                                                                                         ema_alexnet_model, ## undefined
                                                                                                         start_epoch, ## undefined
                                                                                                         best_acc1, ## undefined
                                                                                                         optimizer, ## undefined
                                                                                                         scheduler, ## undefined
                                                                                                         "resume")
        print("Loaded pretrained generator model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))## undefined

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs): ## undefined
        train(alexnet_model, ema_alexnet_model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer) ## undefined
        acc1 = validate(ema_alexnet_model, valid_prefetcher, epoch, writer, "Valid")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1 ## undefined
        is_last = (epoch + 1) == config.epochs
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({"epoch": epoch + 1,
                         "best_acc1": best_acc1,
                         "state_dict": alexnet_model.state_dict(),
                         "ema_state_dict": ema_alexnet_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]: ## Method used to load the images into memory
    # Load train, test and valid datasets
    train_dataset = ImageDataset(config.train_image_dir, config.image_size, "Train") ## Variable used to locate the training set images
    valid_dataset = ImageDataset(config.valid_image_dir, config.image_size, "Valid") ## Variable used to locate the validation set images

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset, ## Training set location given as parameter to the data loader
                                  batch_size=config.batch_size, ## Set the loading batch size as the one declared in config.py
                                  shuffle=True, ## Load images in a randomized order
                                  num_workers=config.num_workers, ## Number of subprocesses used for loading data
                                  pin_memory=True, ## CUDA specific parameter the allow data transfer between CPU and GPU to be faster
                                  drop_last=True, ## Drop incomplete batches (batch size does not match predefined batch size)
                                  persistent_workers=True) ## Batch processing threads are persisted through the epochs
    valid_dataloader = DataLoader(valid_dataset, ## Validation set location gives as parameter to the data loader
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device) ## Initialize the cuda prefetcher for the training set
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device) ## Initialize the cuda prefetcher for the validation set

    return train_prefetcher, valid_prefetcher ## Returns the prefetchers


def build_model() -> [nn.Module, nn.Module]:
    alexnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes) ## undefined
    alexnet_model = alexnet_model.to(device=config.device, memory_format=torch.channels_last) ## undefined

    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_alexnet_model = AveragedModel(alexnet_model, avg_fn=ema_avg) ## undefined

    return alexnet_model, ema_alexnet_model ## undefined


def define_loss() -> nn.CrossEntropyLoss: ## undefined
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing) ## undefined
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last) ## undefined

    return criterion ## undefined


def define_optimizer(model) -> optim.SGD: ## undefined
    optimizer = optim.SGD(model.parameters(), ## undefined
                          lr=config.model_lr, ## undefined
                          momentum=config.model_momentum, ## undefined
                          weight_decay=config.model_weight_decay) ## undefined

    return optimizer ## undefined


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts: ## undefined
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, ## undefined
                                                         config.lr_scheduler_T_0, ## undefined
                                                         config.lr_scheduler_T_mult, ## undefined
                                                         config.lr_scheduler_eta_min) ## undefined

    return scheduler


def train( ## Method to start the training process
        alexnet_model: nn.Module, ## The model to be trained
        ema_model: nn.Module, ## EMA(Exponential Moving Average) used to update the model's parameters
        train_prefetcher: CUDAPrefetcher, ## Used for preloading and prefetching batches of training data
        criterion: nn.CrossEntropyLoss, ## undefined
        optimizer: optim.Adam, ## Optimizer that updates the parameters of the model's predictions
        epoch: int, ## Current epoch
        scaler: amp.GradScaler, ## undefined
        writer: SummaryWriter ## Used for logging and visualization during training
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher) ## Represent the number of batches processed in the current epoch
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f") ## Creates an instance of the AverageMeter class with label Time and the values held to be displayed as 6 characters total, 3 characters for decimal
    data_time = AverageMeter("Data", ":6.3f") ## Creates an instance of the AverageMeter class with label Data and the values held to be displayed as 6 characters total, 3 characters for decimal
    losses = AverageMeter("Loss", ":6.6f") ## Creates an instance of the AverageMeter class with label Loss and the values held to be displayed as 6 characters total, 6 characters for decimal
    acc1 = AverageMeter("Acc@1", ":6.2f") ## Creates an instance of the AverageMeter class with label Acc@1 and the values held to be displayed as 6 characters total, 2 characters for decimal
    acc5 = AverageMeter("Acc@5", ":6.2f") ## Creates an instance of the AverageMeter class with label Acc@5 and the values held to be displayed as 6 characters total, 2 characters for decimal
    progress = ProgressMeter(batches, ## undefined
                             [batch_time, data_time, losses, acc1, acc5], ## undefined
                             prefix=f"Epoch: [{epoch + 1}]") ## undefined

    # Put the generative network model in training mode
    alexnet_model.train() ## Place the model into training mode

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0 ## Initialize batch index to keep track of current batch

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset() ## Reset the prefetcher to start training data from the beginning
    batch_data = train_prefetcher.next() ## Loads the first batch of data using the train prefetcher

    # Get the initialization training time
    end = time.time() ## Records the current time as starting from the first batch

    while batch_data is not None: ## If the images from the batch are not loaded
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end) ## Updates the value of the average class for time with the time taken to load the batch

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True) ## undefined
        target = batch_data["target"].to(device=config.device, non_blocking=True) ## undefined

        # Get batch size
        batch_size = images.size(0) ## Initializez the batch size with the first dimension of the images sensor

        # Initialize generator gradients
        alexnet_model.zero_grad(set_to_none=True) ## undefined

        # Mixed precision training
        with amp.autocast(): ## undefined
            output = alexnet_model(images) ## undefined
            loss = criterion(output, target) ## undefined

        # Backpropagation
        scaler.scale(loss).backward() ## undefined
        # update generator weights
        scaler.step(optimizer) ## undefined
        scaler.update() ## undefined

        # Update EMA
        ema_model.update_parameters(alexnet_model) ## undefined

        # measure accuracy and record loss
        top1, top5 = accuracy(output, target, topk=(1, 5)) ## undefined
        losses.update(loss.item(), batch_size) ## undefined
        acc1.update(top1[0].item(), batch_size) ## undefined
        acc5.update(top5[0].item(), batch_size) ## undefined

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end) ## undefined
        end = time.time() ## undefined

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0: ## undefined
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1) ## undefined
            progress.display(batch_index + 1) ## undefined

        # Preload the next batch of data
        batch_data = train_prefetcher.next() ## undefined

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1 ## undefined


def validate(
        ema_alexnet_model: nn.Module, ## undefined
        data_prefetcher: CUDAPrefetcher, ## undefined
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher) ## undefined
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE) ## undefined
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE) ## undefined
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1, acc5], prefix=f"{mode}: ") ## undefined

    # Put the exponential moving average model in the verification mode
    ema_alexnet_model.eval() ## undefined

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0 ## Sets the starting batch index to 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset() ## Reset the values of the data prefetcher
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad(): ## Disables gradient calculation to improve performance
        while batch_data is not None: ## Ensure loop is going while there is still data in the prefetcher
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True) ## undefined
            target = batch_data["target"].to(device=config.device, non_blocking=True) ## undefined

            # Get batch size
            batch_size = images.size(0) ## Sets the batch_size as the first dimension of the images tensor

            # Inference
            output = ema_alexnet_model(images) ## undefined

            # measure accuracy and record loss
            top1, top5 = accuracy(output, target, topk=(1, 5)) ## Sets the values of the top1 and top5 as the results of the accuracies
            acc1.update(top1[0].item(), batch_size) ## undefined
            acc5.update(top5[0].item(), batch_size) ## undefined

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end) ## undefined
            end = time.time() ## undefined

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0: ## undefined
                progress.display(batch_index + 1) ## undefined

            # Preload the next batch of data
            batch_data = data_prefetcher.next() ## undefined

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1 ## undefined

    # print metrics
    progress.display_summary() ## Prints the summary of the current iteration

    if mode == "Valid" or mode == "Test": ## Checks the mode
        writer.add_scalar(f"{mode}/Acc@1", acc1.avg, epoch + 1) ## undefined
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.") ## Throws error for invalid mode

    return acc1.avg ## undefined


if __name__ == "__main__": ## This is to make sure that the main function is executed only when the script is ran directly and not imported as a module into another script
    main()
