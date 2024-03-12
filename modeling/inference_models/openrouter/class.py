import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)

from modeling.inference_models.openrouter_handler import model_backend as openrouter_handler_model_backend

model_backend_name = "OpenRouter"
model_backend_type = "OpenRouter" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class OpenAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_backend(openrouter_handler_model_backend):
    """InferenceModel for interfacing with OpenRouter's generation API."""
    
    def __init__(self):
        super().__init__()
        self.url = "https://openrouter.ai/api/v1/models" #Due to functionality elsewhere in the code, this needs to be like this. But the actual server is https://openrouter.ai/api/v1/chat
        self.source = "OpenRouter"
    
    def is_valid(self, model_name, model_path, menu_path):
        return model_name == "OpenRouter"