import requests, json
from typing import List, Optional
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
        self.pres_pen = 0

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

    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}, loaded_parameters = {}):

        default_parameters = {'pres_pen': 0}

        if loaded_parameters == {}:
            loaded_parameters = self.read_settings_from_file()

        for key in default_parameters:
            if key not in loaded_parameters:
                loaded_parameters[key] = default_parameters[key]

        self.source = model_name

        requested_parameters = super().get_requested_parameters(model_name, model_path, menu_path, parameters, loaded_parameters)
        requested_parameters.append({
            "uitype": "slider",
            "unit": "float",
            "label": "Presence Penalty",
            "id": "pres_pen",
            "min": -2.0,
            "max": 2.0,
            "step": 0.05,
            "default": loaded_parameters['pres_pen'],
            "tooltip": "(only some models) Adjusts how often the model repeats specific tokens already used in the input. Higher values make such repetition less likely, while negative values do the opposite. Token penalty does not scale with the number of occurrences. Negative values will encourage token reuse.",
            "menu_path": "Configuration",
            "extra_classes": "",
            "refresh_model_inputs": False
        })
        return requested_parameters
    
    def set_input_parameters(self, parameters):
        super().set_input_parameters(parameters)
        self.pres_pen = parameters['pres_pen']

    def _save_settings(self, settings={}):
        settings.update({
                        "pres_pen": self.pres_pen,
                    })
        super()._save_settings(settings)
    
    def is_valid(self, model_name, model_path, menu_path):
        return model_name == "GooseAI"
    
    def get_supported_gen_modes(self) -> List[GenerationMode]:
        return super().get_supported_gen_modes() + [
            GenerationMode.UNTIL_EOS
        ]
    
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