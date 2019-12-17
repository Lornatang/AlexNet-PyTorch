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

import os

import torch
from PIL import Image

from utils.process import pil_to_tensor
from utils.process import tensor_to_pil


def classifier(model, model_path, datasets, dataroot, img_size, channels, classes_names, device):
  """ A classification function used to classify pictures of locations.

  Args:
    model: Modified neural network model.
    model_path: Loaded model address.
    datasets: The folder used for the final classification
    dataroot: Data folders to categorize.
    img_size: Put the size of the image.
    channels: image channel. rgb->3, gray->1.
    classes_names: Categories of objects.
    device: Run-time device selection.

  Examples:
    >> model = Model(**kwargs)
    >> model_path = "alex.pt"
    >> dataset = "mnist"
    >> dataroot = "unknown"
    >> img_size = 28
    >> channels = 1
    >> classes_name = ["1", "2", "3", "4"]
    >> device = torch.device(**kwargs)
    >> classifier(model, model_path, datasets, dataroot, img_size, channels, classes_name, device)
  """
  # create classifier folder.
  for i in range(len(classes_names)):
    try:
      dirname = os.path.join(os.getcwd(), datasets, classes_names[i])
      os.makedirs(dirname)
    except OSError:
      pass

  # load model file
  model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
  model.eval()
  with torch.no_grad():
    for file in os.listdir(dataroot):
      # get image abs path
      img_path = os.path.join(os.getcwd(), dataroot, file)
      # ignore mac sys file
      if os.path.split(img_path)[1] == ".DS_Store":
        continue
      # Load image data and transfrom tensor data.
      image = Image.open(img_path)
      inputs = pil_to_tensor(img_size, channels, image, device)
      inputs = inputs.to(device)

      # prediction image label.
      predicted = model(inputs)
      label = int(torch.argmax(predicted[0]))

      # Put the identified pictures into the corresponding folder.
      image = tensor_to_pil(inputs)
      img_path = os.path.join(datasets, classes_names[label], file)
      image.save(img_path)
