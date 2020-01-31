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

import collections
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
    "avg_size", "classifier_size", "batch_norm_momentum",
    "batch_norm_epsilon", "dropout_rate", "image_size",
    "num_classes"])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def alexnet_params(model_name):
    """ Map AlexNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients: avgpool_size, classifier_size, dropout, res
        "alexnet-a0":  (1,  512, 0.1,  32),
        "alexnet-a1":  (6, 4096, 0.2, 224),
    }
    return params_dict[model_name]


def alexnet(avg_size=None, classifier_size=None, dropout_rate=None, image_size=None, num_classes=1000):
    """ Creates a alexnet model. """

    global_params = GlobalParams(
        avg_size=avg_size,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        classifier_size=classifier_size,
        dropout_rate=dropout_rate,
        image_size=image_size,
        num_classes=num_classes,
    )

    return global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("alexnet"):
        a, c, p, s = alexnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        global_params = alexnet(avg_size=a, classifier_size=c, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError(f"Model name is not pre-defined: {model_name}.")
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return global_params


url_map = {
    "alexnet-a0": "https://github.com/Lornatang/models/raw/master/alexnet/alexnet-a0-8e12ce6b.pth",
    "alexnet-a1": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("classifier.6.weight")
        state_dict.pop("classifier.6.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(
            ["classifier.6.weight", "classifier.6.bias"]), "issue loading pretrained weights"
    print(f"Loaded pretrained weights for {model_name}")
