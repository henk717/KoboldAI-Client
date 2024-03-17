import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os

import utils
from logger import logger
from modeling.inference_model import (
    GenerationSettings,
    ModelCapabilities,
    GenerationMode,
)

from modeling.inference_models.api_handler import(
    model_backend as api_handler_model_backend,
    api_call
)

from modeling.stoppers import Stoppers

model_backend_name = "GooseAI"
model_backend_type = "GooseAI" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class GooseAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_backend(api_handler_model_backend):
    """InferenceModel for interfacing with GooseAI's generation API."""
    
    def __init__(self):
        super().__init__()
        self.url = "https://api.goose.ai/v1/engines"
        self.source = "GooseAI"

        self.post_token_hooks = [
            #PostTokenHooks.stream_tokens,
        ]

        self.stopper_hooks = [
            #Stoppers.core_stopper,
            #Stoppers.dynamic_wi_scanner, #Big nope
            Stoppers.singleline_stopper,
            #Stoppers.chat_mode_stopper, #Implemented by checking chatmode var directly!
            #Stoppers.stop_sequence_stopper,
        ]

        self.capabilties = ModelCapabilities(
            #embedding_manipulation=True,
            #post_token_hooks=True,
            stopper_hooks=True,
            #post_token_probs=True,
        )
        #self._old_stopping_criteria = None
    
    def is_valid(self, model_name, model_path, menu_path):
        return model_name == "GooseAI"
    
    
    
    def get_models(self):
        if self.key == "":
            return []
        
            
        # Get list of models from OAI
        logger.init("OAI Engines", status="Retrieving")
        req = requests.get(
            self.url, 
            headers = {
                'Authorization': 'Bearer '+self.key
                }
            )
        if(req.status_code == 200):
            r = req.json()
            engines = r["data"]
            try:
                engines = [{"value": en["id"], "text": "{} ({})".format(en['id'], "Ready" if en["ready"] == True else "Not Ready")} for en in engines]
            except:
                logger.error(engines)
                raise
            
            online_model = ""

                
            logger.init_ok("OAI Engines", status="OK")
            logger.debug("OAI Engines: {}".format(engines))
            return engines
        else:
            # Something went wrong, print the message and quit since we can't initialize an engine
            logger.init_err("OAI Engines", status="Failed")
            logger.error(req.json())
            emit('from_server', {'cmd': 'errmsg', 'data': req.json()})
            return []
    
    def _raw_api_generate(
        self,
        prompt_plaintext: str,
        max_new: int,
        gen_settings: GenerationSettings,
        batch_count: Optional[int] = 1,
        do_streaming: Optional[bool] = False,
        is_core: Optional[bool] = False,
        seed: Optional[int] = None,
        chatmode: Optional[bool] = False,
        gen_mode: Optional[GenerationMode] = GenerationMode.STANDARD,
        **kwargs,
    ) -> List:

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = prompt_plaintext

        reqdata = {
            "prompt": prompt_plaintext,
            "max_tokens": max_new,
            #min_tokens: 0,
            "temperature": gen_settings.temp,
            "top_a": gen_settings.top_a,
            "top_p": gen_settings.top_p,
            #logit_bias: {},
            "stop": self.plaintext_stoppers,
            "top_k": gen_settings.top_k,
            "tfs": gen_settings.tfs,
            "typical_p": gen_settings.typical,
            #logprobs: false,
            #echo: false,
            "presence_penalty": self.pres_pen,
            "repetition_penalty": gen_settings.rep_pen,
            "repetition_penalty_slope": gen_settings.rep_pen_slope,
            "repetition_penalty_range": gen_settings.rep_pen_range,
            "n": batch_count,
            # TODO: Implement streaming, min tokens, logit_bias, logprobs and maybe also echo option
            "stream": False,
        }

        url= "{}/{}/completions".format(self.url, self.model_name)
        headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            }
        
        call={"url": url, "reqdata": reqdata, "headers": headers}

        item=api_call(call) #Call the API
        outputs = [out["text"] for out in item["choices"]]
        return outputs