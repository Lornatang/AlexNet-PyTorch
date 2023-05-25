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
from enum import Enum

import torch
from torch import nn

__all__ = [
    "accuracy", "load_state_dict", "make_directory", "save_checkpoint", "Summary", "AverageMeter", "ProgressMeter"
]


def accuracy(output, target, topk=(1,)): ## Utility function to compare the output with the target (accuracy) of top k predictions (defaulted to 1)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad(): ## No gradient calculations should be performed (performance boost when only evaluation is needed) 
        maxk = max(topk) ## Max of topk
        batch_size = target.size(0) ## Chooses target's tensor first dimension as batch size

        _, pred = output.topk(maxk, 1, True, True) ## Retrieve top k predicitons of model's output along with their indices. '_' discards the values, indices are stored in 'pred'
        pred = pred.t() ## Transposes the pred tensor
        correct = pred.eq(target.view(1, -1).expand_as(pred)) ## undefined

        results = [] ## Declares an empty array where accuracies will be stored
        for k in topk: ## Iterate over all values of top k predictions
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) ## Selects the first k rows from the 'correct' tensor and reshapes it into a vector. It then calculates the sum of all the values
            results.append(correct_k.mul_(100.0 / batch_size)) ## The correct predictions of k are averaged by the batch size and then added to the resulted accuracies list
        return results ## Return the resulted accuracies


def load_state_dict( ## Defines a method named load_state_dict
        model: nn.Module, ## The model on which the loaded weights will be applied
        model_weights_path: str, ## The path to the file containing the saved model weights and parameters
        ema_model: nn.Module = None, ## Optional for Exponential Moving Average model, defaulted to None
        start_epoch: int = None, ## Optional regarding the epoch on which to start the training process, defaulted to None
        best_acc1: float = None, ## Optional float to describe the best accuracy achieved during training defaulted to None
        optimizer: torch.optim.Optimizer = None, ## Optional optimizer object defaulted to None
        scheduler: torch.optim.lr_scheduler = None, ## Optional learning rate scheduler object defaulted to None
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]: ## The return type of the method
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage) ## undefined

    if load_mode == "resume": ## undefined
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict() ## undefined
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()} ## undefined
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict) ## undefined
        model.load_state_dict(model_state_dict) ## undefined
        # Load ema model state dict. Extract the fitted model weights
        ema_model_state_dict = ema_model.state_dict()
        ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
        # Overwrite the model weights to the current model (ema model)
        ema_model_state_dict.update(ema_state_dict)
        ema_model.load_state_dict(ema_model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict() ## undefined
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if ## undefined
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()} ## undefined
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict) ## undefined
        model.load_state_dict(model_state_dict) ## undefined

    return model, ema_model, start_epoch, best_acc1, optimizer, scheduler


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_checkpoint( ## undefined
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name) ## undefined
    torch.save(state_dict, checkpoint_path) ## undefined

    if is_best: ## undefined
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "best.pth.tar")) ## undefined
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "last.pth.tar")) ## undefined


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object): ##Utility class for computing the average of a value over multiple iterations  
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE): ## Object constructor with a lavbel for the meter, formating as defaulted to floating point number and the summary type
        self.name = name ## Initializes the meter label
        self.fmt = fmt ## Initializes the format type
        self.summary_type = summary_type ## Initializes the summary type
        self.reset() ## Private method to initialize object parameters with default values

    def reset(self): ## Private method to reset the state of the average object
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): ## Method used to update the values the object during iterations
        self.val = val ## Update value with new value
        self.sum += val * n ## Update the sum with the value * number of iterations
        self.count += n ## Add to the number of iterations
        self.avg = self.sum / self.count ## Average of the sum based on number of iterations

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object): ## Declaration of a utlity class for displaying information about training and evaluation process
    def __init__(self, num_batches, meters, prefix=""): ## Object constructor
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches) ## Total number of batches in the training or evaluation process
        self.meters = meters
        self.prefix = prefix

    def display(self, batch): ## Prints progress for current batch
        entries = [self.prefix + self.batch_fmtstr.format(batch)] ## undefined
        entries += [str(meter) for meter in self.meters] ## undefined
        print("\t".join(entries))

    def display_summary(self): ## Method to display the summary of the meters
        entries = [" *"] ## Initializes the entries array
        entries += [meter.summary() for meter in self.meters] ## Concatenates the summary of each meter 
        print(" ".join(entries)) ## Joins the results by empty string and displays the resulted string

    def _get_batch_fmtstr(self, num_batches): ## Private method used in constructor to format a string displaying batch count / total number of batches
        num_digits = len(str(num_batches // 1)) ## Integer divison for num_batches to work for the lengthv of the string
        fmt = "{:" + str(num_digits) + "d}" ## undefined
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" ## Formats string with aggregated data 
