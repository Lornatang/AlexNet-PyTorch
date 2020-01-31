# Copyright 2020 Lorna Authors. All Rights Reserved.
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

import torch
import torch.nn as nn

from .utils import alexnet_params
from .utils import get_model_params
from .utils import load_pretrained_weights


# AlexNet model architecture from the One weird trick...
# <https://arxiv.org/abs/1404.5997>`_ paper.
class AlexNet(nn.Module):
    """ An AlexNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
      global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = AlexNet.from_pretrained("alexnet-e0")
    """

    def __init__(self, global_params=None):
        super().__init__()
        self.avg_size = global_params.avg_size
        self.classifier_size = global_params.classifier_size
        self.dropout_rate = global_params.dropout_rate
        self.image_size = global_params.image_size
        self.num_classes = global_params.num_classes

        def block(in_channels, out_channels, kernel_size=3, padding=1, maxpool=True):
            r""" Define neuron module layer.

            Args:
                in_channels (int): Number of channels in the input image.
                out_channels (int): Number of channels produced by the convolution.
                kernel_size (int or tuple): Size of the convolving kernel.
                stride (int or tuple, optional): Stride of the convolution. Default: 1.
                padding (int or tuple, optional): Zero-padding added to both sides of 
                    the input. Default: 0.
                padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`.
                maxpool (bool, optional): If ``True``, adds a max pool layer to the layers. Default: ``True``

            Returns:
                Some neural model layers 

            Example::

                >>> block(6, 16, 3, 1, 1, maxpool=False)
                [Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 ReLU(inplace=True)]
                >>> block(6, 16, 3, 1, 1)
                [Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 ReLU(inplace=True),
                 MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)]
            """
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
            layers.append(nn.ReLU(inplace=True))
            if maxpool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            return layers

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *block(64, 192, kernel_size=5, padding=2),
            *block(192, 384, maxpool=False),
            *block(384, 256, maxpool=False),
            *block(256, 256),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((self.avg_size, self.avg_size))
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(256 * self.avg_size * self.avg_size, self.classifier_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.classifier_size, self.classifier_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.classifier_size, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        global_params = get_model_params(model_name, override_params)
        return cls(global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={"num_classes": num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000 or num_classes == 10))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, _, res = alexnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (alexnet) at the moment. """
        if model_name not in "alexnet":
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))
