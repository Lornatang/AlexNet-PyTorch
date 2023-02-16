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
from typing import Any

import torch ## undefined
from torch import Tensor ## undefined
from torch import nn ## undefined

__all__ = [
    "AlexNet",
    "alexnet",
]


class AlexNet(nn.Module): ## undefined
    def __init__(self, num_classes: int = 1000) -> None: ## undefined
        super(AlexNet, self).__init__() ## undefined

        self.features = nn.Sequential( ## undefined
            nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)), ## undefined
            nn.ReLU(True), ## undefined
            nn.MaxPool2d((3, 3), (2, 2)), ## undefined

            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)), ## undefined
            nn.ReLU(True), ## undefined
            nn.MaxPool2d((3, 3), (2, 2)), ## undefined

            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)), ## undefined
            nn.ReLU(True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)), ## undefined
            nn.ReLU(True), 
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)), ## undefined
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)), ## undefined
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) ## undefined

        self.classifier = nn.Sequential( ## undefined
            nn.Dropout(0.5), ## undefined
            nn.Linear(256 * 6 * 6, 4096), ## undefined
            nn.ReLU(True), 
            nn.Dropout(0.5), ## undefined
            nn.Linear(4096, 4096), ## undefined
            nn.ReLU(True),
            nn.Linear(4096, num_classes), ## undefined
        )

    def forward(self, x: Tensor) -> Tensor: ## undefined
        return self._forward_impl(x) ## undefined

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor: ## undefined
        out = self.features(x) ## undefined
        out = self.avgpool(out) ## undefined
        out = torch.flatten(out, 1) ## undefined
        out = self.classifier(out) ## undefined

        return out


def alexnet(**kwargs: Any) -> AlexNet: ## undefined
    model = AlexNet(**kwargs) ## undefined

    return model ## undefined
