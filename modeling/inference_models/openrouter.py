import torch
import requests,json
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



class OpenAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_backend(InferenceModel):
    """InferenceModel for interfacing with OpenRouter's generation API."""
    
    def __init__(self):
        super().__init__()
        self.key = ""
        self.url = "https://openrouter.ai/api/v1/chat"
    
    def is_valid(self, model_name, model_path, menu_path):
        return model_name == "OpenRouter"
    
    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        if os.path.exists("settings/{}.model_backend.settings".format(self.source)) and 'colaburl' not in vars(self):
            with open("settings/{}.model_backend.settings".format(self.source), "r") as f:
                try:
                    self.key = json.load(f)['key']
                except:
                    pass
        if 'key' in parameters:
            self.key = parameters['key']
        self.source = model_name
        requested_parameters = []
        requested_parameters.extend([{
                                        "uitype": "text",
                                        "unit": "text",
                                        "label": "Key",
                                        "id": "key",
                                        "default": self.key,
                                        "check": {"value": "", 'check': "!="},
                                        "tooltip": "User Key to use when connecting to OpenRouter.",
                                        "menu_path": "",
                                        "refresh_model_inputs": True,
                                        "extra_classes": ""
                                    },
                                    {
                                        "uitype": "dropdown",
                                        "unit": "text",
                                        "label": "Model",
                                        "id": "model",
                                        "default": "",
                                        "check": {"value": "", 'check': "!="},
                                        "tooltip": "Which model to use when running OpenRouter.",
                                        "menu_path": "",
                                        "refresh_model_inputs": False,
                                        "extra_classes": "",
                                        'children': self.get_oai_models(),

                                    }])
        return requested_parameters
        
    def set_input_parameters(self, parameters):
        self.key = parameters['key'].strip()
        self.model_name = parameters['model']

    def get_oai_models(self):
        if self.key == "":
            return []
        
            
        # Get list of models from OpenRouter
        logger.init("OpenRouter Engines", status="Retrieving")
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
                engines = [{"value": en["id"], "text": "{} ({})".format(en['id'], "Ready")} for en in engines]
            except:
                logger.error(engines)
                raise
            
            online_model = ""

                
            logger.init_ok("OpenRouter Engines", status="OK")
            logger.debug("OpenRouter Engines: {}".format(engines))
            return engines
        else:
            # Something went wrong, print the message and quit since we can't initialize an engine
            logger.init_err("OpenRouter Engines", status="Failed")
            logger.error(req.json())
            emit('from_server', {'cmd': 'errmsg', 'data': req.json()})
            return []
            

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer("gpt2")

    def _save_settings(self):
        with open("settings/{}.model_backend.settings".format(self.source), "w") as f:
            json.dump({"key": self.key}, f, indent="")

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:

        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        # Build request JSON data
        # GooseAI is a subntype of OAI. So to check if it's this type, we check the configname as a workaround
        # as the koboldai_vars.model will always be OAI
        if False:#gen_settings.chat_mode: TODO: Implement chat mode
            promptormessage = "messages"
            payload = {"role": "user", "content": decoded_prompt}

        else:
            promptormessage = "prompt"
            payload = decoded_prompt
        reqdata = {

            #Use one of messages or prompt!
            #messages: Message[],
            promptormessage: payload,
            "model": self.model_name, #Required for OpenRouter!

            # TODO: Implement streaming & stop sequences
            "stop": [],
            "stream": False,

            "max_tokens": max_new,
            "temperature": gen_settings.temp,
            "top_p": gen_settings.top_p,
            "top_k": gen_settings.top_k,
            "frequency_penalty": gen_settings.freq_pen,
            "presence_penalty": gen_settings.pres_pen,
            "repetition_penalty": gen_settings.rep_pen,
            "seed": seed, #OpenAI only
            #"n": batch_count, not an option for openrouter :(
            
            # TODO: Implement logit bias
            "logit_bias": {}
            
        }
        j=[{} for i in range(0,batch_count-1)] ##Fake batching
        for i in range(1, batch_count):
            #req = requests.post(
            #    "{}/completions".format(self.url),
            #    json=reqdata,
            #    headers={
            #        "Authorization": "Bearer " + self.key,
            #        "HTTP-Referer": "https://github.com/scott-ca/KoboldAI-united",
            #        "Content-Type": "application/json"
            #    }
            #)
            
            j[i-1] = req.json()

            if not req.ok:
                # Send error message to web client
                if "error" in j[i]:
                    error_type = j[i]["error"]["type"]
                    error_message = j[i]["error"]["message"]
                else:
                    error_type = "Unknown"
                    error_message = "Unknown"
                raise OpenAIAPIError(error_type, error_message)

        outputs = [out["text"] for response in j for out in response["choices"]]
        return GenerationResult(
            model=self,
            out_batches=np.array([self.tokenizer.encode(x) for x in outputs]),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )
