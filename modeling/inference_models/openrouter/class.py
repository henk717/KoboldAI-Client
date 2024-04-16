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
    batch_api_call
)

from modeling.stoppers import Stoppers

model_backend_name = "OpenRouter"
model_backend_type = "OpenRouter" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class OpenRouterAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_backend(api_handler_model_backend):
    """InferenceModel for interfacing with OpenRouter's generation API."""
    
    def __init__(self):
        super().__init__()
        self.url = "https://openrouter.ai/api/v1/models" #Due to functionality elsewhere in the code, this needs to be like this. But the actual server is https://openrouter.ai/api/v1/chat
        self.serverurl = "https://openrouter.ai/api/v1/chat/completions"
        self.source = "OpenRouter"

        self.post_token_hooks = [
            #PostTokenHooks.stream_tokens,
        ]

        self.stopper_hooks = [
            #Stoppers.core_stopper,
            #Stoppers.dynamic_wi_scanner, #Big nope on OpenRouter
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
        return model_name == "OpenRouter"        
    
    def get_supported_gen_modes(self) -> List[GenerationMode]:
        return super().get_supported_gen_modes() + [
            GenerationMode.UNTIL_EOS
        ]
    
    def get_models(self):
        if self.key == "":
            return []
        
            
        # Get list of models from OpenRouter
        logger.init("OpenRouter Engines", status="Retrieving")
        req = requests.get(
            self.url, 
            headers = {
                'Authorization': 'Bearer '+ self.key
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

        if is_core & utils.koboldai_vars.chatmode: #TODO: Improve chat mode, openrouter permits sending "user" and "bot" names in addition to the messages
            promptormessage = "messages"
            payload = [{"role": "user", "content": prompt_plaintext}]

        else:
            promptormessage = "prompt"
            payload = prompt_plaintext


        reqdata = {
            promptormessage: payload,
            "model": self.model_name, #Required for OpenRouter!

            # TODO: Implement streaming & more stop sequences
            "stop": self.plaintext_stoppers,
            "stream": False,

            "max_tokens": max_new,
            "temperature": gen_settings.temp,
            "top_p": gen_settings.top_p,
            "top_k": gen_settings.top_k,
            "frequency_penalty": utils.koboldai_vars.freq_pen,
            "presence_penalty": utils.koboldai_vars.pres_pen,
            "repetition_penalty": gen_settings.rep_pen,
            "seed": seed, #OpenAI only
            
            # TODO: Implement logit bias
            #"logit_bias": {}
            
        }
        
        url=self.serverurl
        headers={
                "Authorization":"Bearer " + self.key,
                "HTTP-Referer": "https://github.com/henk717/KoboldAI", #For funsies
                "X-Title": "KoboldAI",
                "Content-Type": "application/json"
            }
        
        call={"url": url, "reqdata": reqdata, "headers": headers}

        items=batch_api_call(call, batch_count) #Call the API with the batch of requests

        outputs=[]
        for item in items: #Strip the outer layer of the response, and append the inner layer to the outputs list
            outputs.append(item["choices"][0]["text"]) #We now have a list of the texts [{"textA"}, {"textB"}, {"textC"} etc.]

        return outputs