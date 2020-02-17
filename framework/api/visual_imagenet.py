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
import json
import os
import ssl
import urllib.request

import torch
import torchvision.transforms as transforms
from PIL import Image
from django.shortcuts import render
from rest_framework.views import APIView

from alexnet_pytorch import AlexNet

# unable to download images problem
try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  # Legacy Python that doesn't verify HTTPS certificates by default
  pass
else:
  # Handle target environment that doesn't support HTTPS verification
  ssl._create_default_https_context = _create_unverified_https_context

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AlexNet.from_pretrained("alexnet")
# move the model to GPU for speed if available
model = model.to(device)
# switch to evaluate mode
model.eval()


def preprocess(filename, label):
  input_image = Image.open(filename)

  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = transform(input_image)
  # create a mini-batch as expected by the model
  input_batch = input_tensor.unsqueeze(0)

  labels_map = json.load(open(label))
  labels_map = [labels_map[str(i)] for i in range(1000)]
  return input_batch, labels_map


def index(request):
  r""" Get the image based on the base64 encoding or url address
      and do the pencil style conversion
  Args:
    request: Post request in url.
      - image_code: 64-bit encoding of images.
      - url:        The URL of the image.

  Return:
    Base64 bit encoding of the image.

  Notes:
    Later versions will not contexturn an image's address,
    but instead a base64-bit encoded address
  """

  return render(request, "index.html")


class IMAGENET(APIView):

  @staticmethod
  def get(request):
    """ Get the image based on the base64 encoding or url address

    Args:
      request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.

    Return:
      Base64 bit encoding of the image.

    Notes:
      Later versions will not contexturn an image's address,
      but instead a base64-bit encoded address
    """

    base_path = "static/imagenet"

    try:
      os.makedirs(base_path)
    except OSError:
      pass

    filename = os.path.join(base_path, "imagenet.png")
    if os.path.exists(filename):
      os.remove(filename)

    context = {
      "status_code": 20000
    }
    return render(request, "imagenet.html", context)

  @staticmethod
  def post(request):
    """ Get the image based on the base64 encoding or url address
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.

    Return:
        Base64 bit encoding of the image.

    Notes:
        Later versions will not contexturn an image's address,
        but instead a base64-bit encoded address
    """

    context = None

    # Get the url for the image
    url = request.POST.get("url")
    base_path = "static/imagenet"
    data_path = "data"

    try:
      os.makedirs(base_path)
    except OSError:
      pass

    filename = os.path.join(base_path, "imagenet.png")
    label = os.path.join(data_path, "labels_map.txt")

    image = urllib.request.urlopen(url)
    with open(filename, "wb") as v:
      v.write(image.read())

    image, labels_map = preprocess(filename, label)
    image = image.to(device)

    with torch.no_grad():
      logits = model(image)
    preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

    for idx in preds:
      label = labels_map[idx]
      probability = torch.softmax(logits, dim=1)[0, idx].item() * 100
      probability = str(probability)[:5]

      context = {
        "status_code": 20000,
        "message": "OK",
        "filename": filename,
        "label": label,
        "probability": probability}
    return render(request, "imagenet.html", context)
