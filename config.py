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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model number class
model_num_classes = 200
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "AelexNet_200-TinyImageNet_200"

if mode == "train":
    # Dataset address
<<<<<<< Updated upstream
    train_image_dir = "./data/TinyImagenet_200/train"
    valid_image_dir = "./data/TinyImagenet_200/valid"
=======
    train_image_dir = f"/home/dl/Datasets/Classifier/TinyImagenet_200/train"
    valid_image_dir = "/home/dl/Datasets/Classifier/TinyImagenet_200/valid"
>>>>>>> Stashed changes

    image_size = 224
    batch_size = 128
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_path = ""

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 600

    # Loss parameters
    loss_label_smoothing = 0.1

    # Optimizer parameter
<<<<<<< Updated upstream
    model_lr = 0.5
=======
    model_lr = 0.01
>>>>>>> Stashed changes
    model_momentum = 0.9
    model_weight_decay = 0.0001
    model_ema_decay = 0.99998

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4
    lr_scheduler_T_mult = 1
    lr_scheduler_eta_min = 1e-4

    # How many iterations to print the training result
    print_frequency = 200
