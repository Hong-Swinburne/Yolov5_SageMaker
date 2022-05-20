"""
ModelHandler defines a model handler for load and inference requests
"""
import json
import io
import logging
from PIL import Image

import numpy as np
import torch

class ModelHandler(object):
    def __init__(self):
        self.initialized = False
        self.model = None
        self.image_size = 640 # reduce size=320 for faster inference

    def initialize(self, context):
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        
        logging.info('### Initialize ###')
        weights = f'{model_dir}/best.pt'

        # Load model
        try:
            self.model = torch.hub.load('/home/model-server/yolo5-detection', 'custom', path=weights, source='local')
        except (RuntimeError) as memerr:
            #if re.search("Failed to allocate (.*) Memory", str(memerr), re.IGNORECASE):
            #    logging.error("Memory allocation exception: {}".format(memerr))
            #    raise MemoryError
            raise

    def preprocess(self, request):
        img_list = []
        for data in enumerate(request):
            # Read the bytearray of the image from the input
            image_file = data.get("body")
            #image_bytes = image_file.read()
            img = Image.open(io.BytesIO(image_file))
            img_list.append(img)

        return img_list
    
    def inference(self, model_input):
        logging.info("### inference - 1")
        logging.info(model_input)
        logging.info(len(model_input))
        inferred = self.model(model_input, size=self.image_size)
        logging.info("### inference - 2")
        logging.info(inferred)
        json_str = inferred.pandas().xyxy[0].to_json(orient="records")
        json_dict = json.loads(json_str)
        results = []
        results.append(json_dict)
        logging.info("### inference - 3")
        logging.info(results)
        return results


    def handle(self, data):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return model_out


_service = ModelHandler()


def handle(data, context):    
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)