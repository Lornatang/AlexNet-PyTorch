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
import csv
import os

from PIL import Image

train_csv_path = "../data/MiniImageNet_1K/original/train.csv"
valid_csv_path = "../data/MiniImageNet_1K/original/valid.csv"
test_csv_path = "../data/MiniImageNet_1K/original/test.csv"

inputs_images_dir = "../data/MiniImageNet_1K/original/mini_imagenet/images"
output_images_dir = "../data/MiniImageNet_1K/"

train_label = {}
val_label = {}
test_label = {}
with open(train_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[0]] = row[1]

with open(valid_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[0]] = row[1]

with open(test_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        test_label[row[0]] = row[1]

for png in os.listdir(inputs_images_dir):
    path = inputs_images_dir + "/" + png
    im = Image.open(path)
    if png in train_label.keys():
        tmp = train_label[png]
        temp_path = output_images_dir + "/train" + "/" + tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t)

    elif png in val_label.keys():
        tmp = val_label[png]
        temp_path = output_images_dir + "/valid" + "/" + tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t)

    elif png in test_label.keys():
        tmp = test_label[png]
        temp_path = output_images_dir + "/test" + "/" + tmp
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        t = temp_path + "/" + png
        im.save(t)
