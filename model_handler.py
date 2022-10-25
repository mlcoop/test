import io
import os
import base64
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import list_classes_from_module
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        self.device = 'cpu'
        self.manifest = context.manifest
        properties = context.system_properties
        serialized_file = self.manifest["model"]["serializedFile"]
        model_dir = properties.get("model_dir")
        model_pt_path = os.path.join(model_dir, serialized_file)
        
        # model_file = self.manifest["model"].get("modelFile", "")
        # module = importlib.import_module(model_file.split(".")[0])
        # model_class_definitions = list_classes_from_module(module)
        # model_class = model_class_definitions[0]

        # self.model = model_class()
        # self.model.load_state_dict(torch.load(model_pt_path, map_location=torch.device('cpu')))
        # self.model = self.model.to(torch.device('cpu'))
        self.model = torch.jit.load(model_pt_path, map_location=torch.device('cpu'))
        self.model = self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        preprocessed_data = []
        for row in data:
            img = row['image'].decode()
            decoded_img = self.decode_img(img)
            pill_img = self.convert(decoded_img)
            transformed_img = self.transform(28, 28)(pill_img)
            preprocessed_data.append(transformed_img)
        return torch.stack(preprocessed_data)

    def postprocess(self, data):
        return data.argmax(dim=1).tolist()

    def decode_img(self, image_file):
        return base64.b64decode(image_file)
    
    def convert(self, image_data):
        return Image.open(io.BytesIO(image_data)).convert('L')
    
    def transform(self, height, width):
        return transforms.Compose([    
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ])
