#Okay, so this whole thing is a bit of an abomination, overwriting the core_generate function from the standard inference model in order to tie into the standard functionality that the rest of the codebase expects. 
#It makes the API interact properly with a fair few features, which I thought was epic. Several parts might also be possible to split out into a more generalized API backend, but that's a task for another day.

from multiprocessing.dummy import Pool
import torch
import requests,json
import numpy as np
import time
from typing import List, Optional, Union
import os
from enum import Enum

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
    use_core_manipulations,
    ModelCapabilities,
)

from modeling.stoppers import Stoppers

class GenerationMode(Enum): #Is this even needed here? Could the enum in inference_model be used directly instead?
    STANDARD = "standard"
    FOREVER = "forever"
    UNTIL_EOS = "until_eos"
    UNTIL_NEWLINE = "until_newline"
    UNTIL_SENTENCE_END = "until_sentence_end"

def stopper_translator(stopper)->str: #This is a bit of a hack, but it makes some simple stoppers available to use in plaintext via api-calls
        if stopper == Stoppers.newline_stopper:
            return "\n"
        elif stopper == Stoppers.sentence_end_stopper:
            return "."
        else:
            return ""

class OpenRouterAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class model_backend(InferenceModel):
    """InferenceModel for interfacing with OpenRouter's generation API."""
    
    def __init__(self):
        super().__init__()
        self.key = ""
        self.url = "https://openrouter.ai/api/v1/models"
        self.pres_pen = 0
        self.freq_pen = 0
        self.pad_token_id= -1

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
    
    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        saved_data = {'key': "", 'freq_pen': 0, 'pres_pen': 0}
        if os.path.exists("settings/{}.model_backend.settings".format(self.source)) and 'colaburl' not in vars(self):
            with open("settings/{}.model_backend.settings".format(self.source), "r") as f:
                try:
                    temp = json.load(f)
                    for key in temp:
                        saved_data[key] = temp[key]
                except:
                    pass
        if 'key' in saved_data:
            self.key = saved_data['key']
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
        requested_parameters.append({ #Could these two be in the regular settings menu somehow?
            "uitype": "slider",
            "unit": "float",
            "label": "Frequency Penalty",
            "id": "freq_pen",
            "min": -2.0,
            "max": 2.0,
            "step": 0.05,
            "default": saved_data['freq_pen'],
            "tooltip": "(only some models) This setting aims to control the repetition of tokens based on how often they appear in the input. It tries to use less frequently those tokens that appear more in the input, proportional to how frequently they occur. Token penalty scales with the number of occurrences. Negative values will encourage token reuse.",
            "menu_path": "Configuration",
            "extra_classes": "",
            "refresh_model_inputs": False
        })
        requested_parameters.append({
            "uitype": "slider",
            "unit": "float",
            "label": "Presence Penalty",
            "id": "pres_pen",
            "min": -2.0,
            "max": 2.0,
            "step": 0.05,
            "default": saved_data['pres_pen'],
            "tooltip": "(only some models) Adjusts how often the model repeats specific tokens already used in the input. Higher values make such repetition less likely, while negative values do the opposite. Token penalty does not scale with the number of occurrences. Negative values will encourage token reuse.",
            "menu_path": "Configuration",
            "extra_classes": "",
            "refresh_model_inputs": False
        })
        return requested_parameters
        
    def set_input_parameters(self, parameters):
        self.key = parameters['key'].strip()
        self.model_name = parameters['model']
        self.pres_pen = parameters['pres_pen']
        self.freq_pen = parameters['freq_pen']
        self.plaintext_stoppers = []

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
            json.dump(
                    {
                        "key": self.key,
                        "pres_pen": self.pres_pen,
                        "freq_pen": self.freq_pen
                    }, 
                    f,
                    indent=""
                )

    
    def get_supported_gen_modes(self) -> List[GenerationMode]:
        return super().get_supported_gen_modes() + [
            GenerationMode.UNTIL_EOS
        ]

    def core_generate( #Overwriting the core_generate function with a copy and hacking it to work with API instead. Lots of the code here could *probably* be thrown our entirely, but I'm not entirely sure what it all does.
        self,
        text: list,
        found_entries: set,
        gen_mode: GenerationMode = GenerationMode.STANDARD,
    ):
        """Generate story text. Heavily tied to story-specific parameters; if
        you are making a new generation-based feature, consider `generate_raw()`.

        Args:
            text (list): Encoded input tokens
            found_entries (set): Entries found for Dynamic WI
            gen_mode (GenerationMode): The GenerationMode to pass to raw_generate. Defaults to GenerationMode.STANDARD

        Raises:
            RuntimeError: if inconsistancies are detected with the internal state and Lua state -- sanity check
            RuntimeError: if inconsistancies are detected with the internal state and core stopper -- sanity check
        """

        start_time = time.time()
        if isinstance(text, torch.Tensor):
            prompt_tokens = text.cpu().numpy()
        elif isinstance(text, list):
            prompt_tokens = np.array(text)
        elif isinstance(text, str):
            prompt_tokens = np.array(self.tokenizer.encode(text))
        else:
            raise ValueError(f"Prompt is {type(text)}. Not a fan!")
        gen_in = torch.tensor(text, dtype=torch.long)[None]
        logger.debug(
            "core_generate: torch.tensor time {}s".format(time.time() - start_time)
        )

        if (
            gen_in.shape[-1] + utils.koboldai_vars.genamt
            > utils.koboldai_vars.max_length
        ):
            logger.error("gen_in.shape[-1]: {}".format(gen_in.shape[-1]))
            logger.error(
                "utils.koboldai_vars.genamt: {}".format(utils.koboldai_vars.genamt)
            )
            logger.error(
                "utils.koboldai_vars.max_length: {}".format(
                    utils.koboldai_vars.max_length
                )
            )
        assert (
            gen_in.shape[-1] + utils.koboldai_vars.genamt
            <= utils.koboldai_vars.max_length
        )

        found_entries = found_entries or set()

        self.gen_state["wi_scanner_excluded_keys"] = found_entries

        utils.koboldai_vars._prompt = utils.koboldai_vars.prompt

        with torch.no_grad():
            already_generated = 0
            numseqs = utils.koboldai_vars.numseqs
            total_gens = None
            start_time = time.time()

            result = self.raw_generate(
                gen_in[0],
                max_new=utils.koboldai_vars.genamt,
                do_streaming=utils.koboldai_vars.output_streaming,
                batch_count=numseqs, #always sending numseqs, because alt_gen seems to never be true?
                is_core=True,
                seed=utils.koboldai_vars.seed
                if utils.koboldai_vars.full_determinism
                else None,
                gen_mode=gen_mode,
            )
            logger.debug(
                "core_api_generate: run raw_generate pass {} {}s".format(
                    already_generated, time.time() - start_time
                )
            )
            
            genout = result.encoded

            already_generated += len(genout[0])

            try:
                assert (
                    already_generated
                    <= utils.koboldai_vars.genamt * utils.koboldai_vars.numseqs
                )
            except AssertionError:
                print("AlreadyGenerated", already_generated)
                print("genamt", utils.koboldai_vars.genamt)
                raise




            if total_gens is None:
                total_gens = genout
            else:
                total_gens = torch.cat((total_gens, genout))

        return total_gens, already_generated


    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        batch_count: Optional[int] = 1,
        do_streaming: Optional[bool] = False,
        is_core: Optional[bool] = False,
        seed: Optional[int] = None,
        chatmode: Optional[bool] = False,
        gen_mode: Optional[GenerationMode] = GenerationMode.STANDARD,
        **kwargs,
    ) -> GenerationResult:

        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        if is_core & utils.koboldai_vars.chatmode: #TODO: Improve chat mode, openrouter permits sending "user" and "bot" names in addition to the messages
            promptormessage = "messages"
            payload = [{"role": "user", "content": decoded_prompt}]

        else:
            promptormessage = "prompt"
            payload = decoded_prompt


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
            "frequency_penalty": self.freq_pen,
            "presence_penalty": self.pres_pen,
            "repetition_penalty": gen_settings.rep_pen,
            "seed": seed, #OpenAI only
            #"n": batch_count, not an option for openrouter :(
            
            # TODO: Implement logit bias
            #"logit_bias": {}
            
        }

        outputs=[]
        items=[None]*batch_count

        def api_caller(reqdata, items, i): #Define a function to call the API, so we can multithread it for more SPEED, just don't accidentally get your key banned for DDOS or something
            if not utils.koboldai_vars.quiet:
                print("Worker", i, "calling API")

            response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", #needs to be hardcoded due to functionality elsewhere in the code
            json=reqdata,
            headers={
                "Authorization":"Bearer " + self.key,
                "HTTP-Referer": "https://github.com/scott-ca/KoboldAI-united", #For funsies
                "Content-Type": "application/json"
            }
            )
            if not utils.koboldai_vars.quiet:
                print(response)
            
            if not response.ok: ##TODO: Check whether this error handling actually works. Dubious
                # Send error message to web client
                if "error" in response:
                    error_type = response["error"]["type"]
                    error_message = response["error"]["message"]
                else:
                    error_type = "Unknown"
                    error_message = "Unknown"
                raise OpenRouterAPIError(error_type, error_message)
            items[i] = response.json()["choices"]

        pool = Pool(batch_count)
        for i in range(batch_count):
            pool.apply_async(api_caller, args=(reqdata, items, i))

        pool.close()
        pool.join() #Pool.join() waits for all the processes to finish, and then the program continues. This is needed to ensure that the program doesn't continue before the processes are finished, as that would cause a crash.
        #Be aware that if an API request fails, we might get stuck here indefinitely. TODO: This is a potential issue that should be addressed in the future.

        for item in items: #Strip the outer layer of the response, and append the inner layer to the outputs list
            if item is None:
                raise OpenRouterAPIError("Unknown", "API call failed: No response received.")
            outputs.append(item[0]["text"]) #We now have a list of the texts [{"textA"}, {"textB"}, {"textC"} etc.]

        tokenized = []
        for x in outputs: #Tokenize the outputs so they can be returned as a GenerationResult
            tokenized.append( self.tokenizer.encode(x))
            max_len = max([len(unit) for unit in tokenized]) #Store the length of the longest tokenized output

        for i in range(len(tokenized)): #homogenize the length of the tokenized outputs so numpy doesn't get mad
            if len(tokenized[i]) < max_len:
                new_unit = [self.pad_token_id for i in range(max_len)]
                new_unit[:len(tokenized[i])] = tokenized[i]
                tokenized[i] = new_unit
    
        return GenerationResult(
            model=self,
            out_batches=np.array(tokenized),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=gen_mode==GenerationMode.UNTIL_SENTENCE_END,
        )

    def raw_generate(
        self,
        # prompt is either a string (text) or a list (token ids)
        prompt: Union[str, list, np.ndarray],
        max_new: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        batch_count: int = 1,
        bypass_hf_maxlength: bool = False,
        generation_settings: Optional[dict] = None,
        is_core: bool = False,
        single_line: bool = False,
        found_entries: set = (),
        tpu_dynamic_inference: bool = False,
        seed: Optional[int] = None,
        gen_mode: GenerationMode = GenerationMode.STANDARD,
        **kwargs,
    ) -> GenerationResult:

        if gen_mode not in self.get_supported_gen_modes():
            gen_mode = GenerationMode.STANDARD
            logger.warning(f"User requested unsupported GenerationMode '{gen_mode}'! Defaulting to STANDARD.")

        self.plaintext_stoppers = [] #API can't take lambda functions, so we need to convert the stoppers to plaintext to include in the api call
        temp_stoppers = []
        if gen_mode == GenerationMode.UNTIL_NEWLINE:
            # TODO: Look into replacing `single_line` with `generation_mode`
            temp_stoppers.append(Stoppers.newline_stopper)
        elif gen_mode == GenerationMode.UNTIL_SENTENCE_END:
            temp_stoppers.append(Stoppers.sentence_end_stopper)

        self.stopper_hooks += temp_stoppers
        for stopper in self.stopper_hooks:
            as_plaintext = stopper_translator(stopper)
            if as_plaintext != "":
                self.plaintext_stoppers.append(as_plaintext)

        utils.koboldai_vars.inference_config.do_core = is_core
        gen_settings = GenerationSettings(*(generation_settings or {}))

        if isinstance(prompt, torch.Tensor):
            prompt_tokens = prompt.cpu().numpy()
        elif isinstance(prompt, list):
            prompt_tokens = np.array(prompt)
        elif isinstance(prompt, str):
            prompt_tokens = np.array(self.tokenizer.encode(prompt))
        else:
            raise ValueError(f"Prompt is {type(prompt)}. Not a fan!")

        time_start = time.time()

        with use_core_manipulations():
            result = self._raw_generate(
                prompt_tokens=prompt_tokens,
                max_new=max_new,
                batch_count=batch_count,
                gen_settings=gen_settings,
                gen_mode=gen_mode,
                seed=seed,
            )

        time_end = round(time.time() - time_start, 2)

        try:
            tokens_per_second = round(len(result.encoded[0]) / time_end, 2)
        except ZeroDivisionError:
            tokens_per_second = 0

        if not utils.koboldai_vars.quiet:
            logger.info(
                f"Generated {len(result.encoded[0])} tokens in {time_end} seconds, for an average rate of {tokens_per_second} tokens per second."
            )

        for stopper in temp_stoppers:
            self.stopper_hooks.remove(stopper)

        return result
