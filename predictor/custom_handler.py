
import os
import json
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler):
    """
    The handler takes an input string and returns the classification text 
    based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        # serialized_file = self.manifest["model"]["serializedFile"]
        # model_pt_path = os.path.join(model_dir, serialized_file)
        # if not os.path.isfile(model_pt_path):
        #     raise RuntimeError("Missing the model.pt or pytorch_model.bin file")
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
        
        # Ensure to use the same tokenizer used during training
        self.tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")

        self.initialized = True

    def preprocess(self, data):
        """ Preprocessing input request by tokenizing
            Extend with your own preprocessing steps as needed
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        # Tokenize the texts
        tokenizer_args = ((sentences,))
        inputs = self.tokenizer(*tokenizer_args,
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        return inputs

    def inference(self, inputs):
        """ Predict the class of a text using a trained transformer model.
        """
        corrections = self.model.generate(**inputs)
        corrections = self.tokenizer.batch_decode(corrections, skip_special_tokens=True)

        for correction in corrections:
            logger.info("Suggested phrase: '%s'", correction)

        return [corrections]

    def postprocess(self, inference_output):
        return inference_output
