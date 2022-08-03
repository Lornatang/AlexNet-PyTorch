from tiyaro.sdk.base_handler import TiyaroBase

import json
import base64
import io

import torch
import torchvision.transforms as transforms
from PIL import Image

from alexnet_pytorch import AlexNet

# Handler for this model was developed by referring to examples/simple/test.py
class TiyaroHandler(TiyaroBase):
    # For image-classification, simply adhere to the input and output format specified in __pre_process() and __post_process(). This will automatically do the following.
    #
    # 1. Tiyaro will automatically generate an OpenAPI spec for your Model's API
    # 2. Tiyaro will automatically generate sample code snippets
    # 3. Tiyaro will automatically provide Demo Screen in the Model Card of your model, to show case live demo instantly to the world
    # 4. Tiyaro will enable you, and your model users, to create experiments and compare with wide range of models in image-classification class in Tiyaro
    # 5. With Tiyaro experiments you and your model users will be able to get comprehensive comparision with other models on various metrics, graphs, etc.,

    def setup_model(self, pretrained_file_path):
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.labels_map = None

        model = AlexNet.from_pretrained("alexnet")
        model.eval()
        model = model.to(self.device)
        self.model = model

        # Load class names
        labels_map = json.load(open("examples/simple/labels_map.txt"))
        labels_map = [labels_map[str(i)] for i in range(1000)]
        self.labels_map = labels_map

    def __pre_process(self, img_bytes):
        # Do the necessary preprocessing
        img = Image.open(io.BytesIO(img_bytes))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        input_batch = input_batch.to(self.device)
        return input_batch

    def infer(self, img_bytes):
        # Example of inference
        input = self.__pre_process(img_bytes)

        with torch.no_grad():
            logits = self.model(input)
            preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

        self.logits = logits
        return self.__post_process(preds)
        
    def __post_process(self, model_output):
        # Example of post-processing
        preds = model_output
        result = {}
        for idx in preds:
            label = self.labels_map[idx]
            prob = torch.softmax(self.logits, dim=1)[0, idx].item()
            result[label] = prob
        return result