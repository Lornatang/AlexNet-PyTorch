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


def accuracy(output, target, topk=(1,)): ## undefined
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad(): ## undefined
        maxk = max(topk) ## undefined
        batch_size = target.size(0) ## undefined

        _, pred = output.topk(maxk, 1, True, True) ## undefined
        pred = pred.t() ## undefined
        correct = pred.eq(target.view(1, -1).expand_as(pred)) ## undefined

        results = [] ## undefined
        for k in topk: ## undefined
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) ## undefined
            results.append(correct_k.mul_(100.0 / batch_size)) ## undefined
        return results ## undefined


def load_state_dict( ## undefined
        model: nn.Module, ## undefined
        model_weights_path: str, ## undefined
        ema_model: nn.Module = None, ## undefined
        start_epoch: int = None, ## undefined
        best_acc1: float = None, ## undefined
        optimizer: torch.optim.Optimizer = None, ## undefined
        scheduler: torch.optim.lr_scheduler = None, ## undefined
        load_mode: str = None,
) -> [nn.Module, nn.Module, str, int, float, torch.optim.Optimizer, torch.optim.lr_scheduler]: ## undefined
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


class AverageMeter(object): ## undefined
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE): ## undefined
        self.name = name ## undefined
        self.fmt = fmt ## undefined
        self.summary_type = summary_type ## undefined
        self.reset() ## undefined

    def reset(self): ## undefined
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): ## undefined
        self.val = val ## undefined
        self.sum += val * n ## undefined
        self.count += n ## undefined
        self.avg = self.sum / self.count ## undefined

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


class ProgressMeter(object): ## undefined
    def __init__(self, num_batches, meters, prefix=""): ## undefined
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches) ## undefined
        self.meters = meters
        self.prefix = prefix

    def display(self, batch): ## undefined
        entries = [self.prefix + self.batch_fmtstr.format(batch)] ## undefined
        entries += [str(meter) for meter in self.meters] ## undefined
        print("\t".join(entries))

    def display_summary(self): ## undefined
        entries = [" *"] ## undefined
        entries += [meter.summary() for meter in self.meters] ## undefined
        print(" ".join(entries)) ## undefined

    def _get_batch_fmtstr(self, num_batches): ## undefined
        num_digits = len(str(num_batches // 1)) ## undefined
        fmt = "{:" + str(num_digits) + "d}" ## undefined
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" ## undefined
