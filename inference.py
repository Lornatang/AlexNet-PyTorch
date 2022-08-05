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
import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import imgproc
from model import alexnet
from utils import load_state_dict


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_num_classes: int, device: torch.device) -> [nn.Module, nn.Module]:
    model = alexnet(model_num_classes)
    model = model.to(device=device, memory_format=torch.channels_last)

    ema_model = AveragedModel(model)

    return model, ema_model


def preprocess_image(image_path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(image_path)

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OpenCV convert PIL
    image = Image.fromarray(image)

    # Resize to 224
    image = Resize([image_size, image_size])(image)
    # Convert image data to pytorch format data
    tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)
    # Convert a tensor image to the given ``dtype`` and scale the values accordingly
    tensor = ConvertImageDtype(torch.float)(tensor)
    # Normalize a tensor image with mean and standard deviation.
    tensor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)

    device = choice_device(args.device_type)

    # Initialize the model
    alexnet, ema_alexnet = build_model(args.model_num_classes, device)
    print("Build AlexNet model successfully.")

    # Load model weights
    alexnet, _, _, _, _, _ = load_state_dict(alexnet, ema_alexnet, args.model_weights_path)
    print(f"Load AlexNet model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    alexnet.eval()

    tensor = preprocess_image(args.image_path, args.image_size, device)

    # Inference
    with torch.no_grad():
        output = alexnet(tensor)

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()

    # Print classification results
    for class_index in prediction_class_index:
        prediction_class_label = class_label_map[class_index]
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")
    parser.add_argument("--model_num_classes", type=int, default=1000)
    parser.add_argument("--model_weights_path", type=str, default="./results/pretrained_models/AlexNet-ImageNet_1K-9df8cd0f.pth.tar")
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device_type", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    main()
