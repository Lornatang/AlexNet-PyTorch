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
import queue ## undefined
import sys ## undefined
import threading ## undefined
from glob import glob ## undefined

import cv2 ## undefined
import torch ## undefined
from PIL import Image ## undefined
from torch.utils.data import Dataset, DataLoader ## undefined
from torchvision import transforms ## undefined
from torchvision.datasets.folder import find_classes ## undefined
from torchvision.transforms import TrivialAugmentWide ## undefined

import imgproc

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32": ## undefined
    delimiter = "\\" ## undefined
else: ## undefined
    delimiter = "/" ## undefined


class ImageDataset(Dataset): ## undefined
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
        super(ImageDataset, self).__init__() ## undefined
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)
        self.image_size = image_size ## undefined
        self.mode = mode ## undefined
        self.delimiter = delimiter## undefined

        if self.mode == "Train": ## undefined
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## undefined
                transforms.RandomResizedCrop(self.image_size), ## undefined
                TrivialAugmentWide(), ## undefined
                transforms.RandomRotation([0, 270]), ## undefined
                transforms.RandomHorizontalFlip(0.5), ## undefined
                transforms.RandomVerticalFlip(0.5), ## undefined
            ])
        elif self.mode == "Valid" or self.mode == "Test": ## undefined
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([ ## undefined
                transforms.Resize(256), ## undefined
                transforms.CenterCrop([self.image_size, self.image_size]), ## undefined
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([ ## undefined
            transforms.ConvertImageDtype(torch.float), ## undefined
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ## undefined
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]: ## undefined
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS: ## undefined
            image = cv2.imread(self.image_file_paths[batch_index]) ## undefined
            target = self.class_to_idx[image_dir] ## undefined
        else: ## undefined
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ## undefined

        # OpenCV convert PIL
        image = Image.fromarray(image) ## undefined

        # Data preprocess
        image = self.pre_transform(image) ## undefined

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False) ## undefined

        # Data postprocess
        tensor = self.post_transform(tensor) ## undefined

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.image_file_paths) ## undefined


class PrefetchGenerator(threading.Thread): ## undefined
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue) ## undefined
        self.generator = generator ## undefined
        self.daemon = True ## undefined
        self.start() ## undefined

    def run(self) -> None: ## undefined
        for item in self.generator: ## undefined
            self.queue.put(item) ## undefined
        self.queue.put(None) ## undefined

    def __next__(self): ## undefined
        next_item = self.queue.get() ## undefined
        if next_item is None: ## undefined
            raise StopIteration ## undefined
        return next_item ## undefined

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader): ## undefined
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None: ## undefined
        self.num_data_prefetch_queue = num_data_prefetch_queue ## undefined
        super(PrefetchDataLoader, self).__init__(**kwargs) ## undefined

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher: ## undefined
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None: ## undefined
        self.original_dataloader = dataloader ## undefined
        self.data = iter(dataloader) ## undefined

    def next(self): ## undefined
        try:
            return next(self.data) ## undefined
        except StopIteration: ## undefined
            return None

    def reset(self): ## undefined
        self.data = iter(self.original_dataloader) ## undefined

    def __len__(self) -> int: ## undefined
        return len(self.original_dataloader) ## undefined


class CUDAPrefetcher: ## undefined
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device): ## undefined
        self.batch_data = None
        self.original_dataloader = dataloader ## undefined
        self.device = device ## undefined

        self.data = iter(dataloader) ## undefined
        self.stream = torch.cuda.Stream() ## undefined
        self.preload() ## undefined

    def preload(self): ## undefined
        try:
            self.batch_data = next(self.data) ## undefined
        except StopIteration: ## undefined
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)
 
    def next(self): ## undefined
        torch.cuda.current_stream().wait_stream(self.stream) ## undefined
        batch_data = self.batch_data ## undefined
        self.preload() ## undefined
        return batch_data ## undefined

    def reset(self): ## undefined
        self.data = iter(self.original_dataloader) ## undefined
        self.preload() ## undefined

    def __len__(self) -> int: ## undefined
        return len(self.original_dataloader) ## undefined
