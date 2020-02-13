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
  """ An AlexNet model. Most easily loaded with the .from_name or
  .from_pretrained methods

  Args:
    global_params (namedtuple): A set of GlobalParams shared between blocks

  Example:
      model = AlexNet.from_pretrained("alexnet")
  """

  def __init__(self, global_params=None):
    super().__init__()
    self._global_params = global_params

    dropout_rate = self._global_params.dropout_rate
    num_classes = self._global_params.num_classes

    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      nn.Dropout(p=dropout_rate),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(p=dropout_rate),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )

  def extract_features(self, inputs):
    """ Returns output of the final convolution layer """
    x = self.features(inputs)
    return x

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
    load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
    return model

  @classmethod
  def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, res = alexnet_params(model_name)
    return res

  @classmethod
  def _check_model_name_is_valid(cls, model_name):
    """ Validates model name. None that pretrained weights are only available for
    the first four models (alexnet) at the moment. """
    valid_model = "alexnet"
    if model_name not in valid_model:
      raise ValueError("model_name should be one of: " + ", ".join(valid_model))
