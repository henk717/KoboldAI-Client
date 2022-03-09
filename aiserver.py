#!/usr/bin/python3
# ==================================================================#
# KoboldAI
# Version: 1.17.0
# By: KoboldAIDev and the KoboldAI Community
# ==================================================================#

import argparse
import bisect
import collections
import contextlib
import gc
import html
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import zipfile
from collections.abc import Iterable
from os import path, getcwd
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List

import bleach
# External packages
import eventlet
import lupa
import markdown
import packaging
import requests
from eventlet import tpool
from flask import Flask, render_template, Response, request, copy_current_request_context
from flask_socketio import SocketIO, emit

# KoboldAI
import fileops
import gensettings
import structures
import utils
from utils import debounce

# package settings
eventlet.monkey_patch(all=True, thread=False)
os.system("")
os.environ['EVENTLET_THREADPOOL_SIZE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if lupa.LUA_VERSION[:2] != (5, 4):
    print(f"Please install lupa==1.10. You have lupa {lupa.__version__}.", file=sys.stderr)


# ==================================================================#
# Variables & Storage
# ==================================================================#

# Terminal tags for colored text
class Colors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    UNDERLINE = '\033[4m'


# AI models
mainmenu = [
    ["Load a model from its directory", "NeoCustom", ""],
    ["Load an old GPT-2 model (eg CloverEdition)", "GPT2Custom", ""],
    ["Skein 6B (Hybrid)", "KoboldAI/GPT-J-6B-Skein", "16GB"],
    ["Janeway 6B (Novel)", "KoboldAI/GPT-J-6B-Janeway", "16GB"],
    ["Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "16GB"],
    ["Lit 6B (NSFW)", "hakurei/lit-6B", "16GB"],
    ["Shinen 6B (NSFW)", "KoboldAI/GPT-J-6B-Shinen", "16GB"],
    ["C1 6B (Chatbot)", "hakurei/c1-6B", "16GB"],
    ["Janeway Neo 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Janeway", "8GB"],
    ["Janeway FSD 2.7B (Novel)", "KoboldAI/fairseq-dense-2.7B-Janeway", "8GB"],
    ["Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "8GB"],
    ["Picard 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Picard", "8GB"],
    ["Horni 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Horni", "8GB"],
    ["Horni-LN 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Horni-LN", "8GB"],
    ["Shinen 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Shinen", "8GB"],
    ["Untuned GPT-Neo/J", "gptneolist", ""],
    ["Untuned Fairseq Dense", "fsdlist", ""],
    ["Untuned XGLM", "xglmlist", ""],
    ["Untuned GPT2", "gpt2list", ""],
    ["Online Services", "apilist", ""],
    ["Read Only (No AI)", "ReadOnly", ""]
]

gptneolist = [
    ["GPT-J 6B", "EleutherAI/gpt-j-6B", "16GB"],
    ["GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "8GB"],
    ["GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "6GB"],
    ["Return to Main Menu", "Return", ""],
]

gpt2list = [
    ["GPT-2 XL", "gpt2-xl", "6GB"],
    ["GPT-2 Large", "gpt2-large", "4GB"],
    ["GPT-2 Med", "gpt2-medium", "2GB"],
    ["GPT-2", "gpt2", "2GB"],
    ["Return to Main Menu", "Return", ""],
]

fsdlist = [
    ["Fairseq Dense 13B", "KoboldAI/fairseq-dense-13B", "32GB"],
    ["Fairseq Dense 6.7B", "KoboldAI/fairseq-dense-6.7B", "16GB"],
    ["Fairseq Dense 2.7B", "KoboldAI/fairseq-dense-2.7B", "8GB"],
    ["Fairseq Dense 1.3B", "KoboldAI/fairseq-dense-1.3B", "6GB"],
    ["Fairseq Dense 355M", "KoboldAI/fairseq-dense-355M", ""],
    ["Fairseq Dense 125M", "KoboldAI/fairseq-dense-125M", ""],
    ["Return to Main Menu", "Return", ""],
]

xglmlist = [
    ["XGLM 4.5B (Larger Dataset)", "facebook/xglm-4.5B", ""],
    ["XGLM 7.5B", "facebook/xglm-7.5B", ""],
    ["XGLM 2.9B", "facebook/xglm-2.9B", ""],
    ["XGLM 1.7B", "facebook/xglm-1.7B", ""],
    ["XGLM 564M", "facebook/xglm-564M", ""],
    ["Return to Main Menu", "Return", ""],
]

apilist = [
    ["GooseAI API (requires API key)", "GooseAI", ""],
    ["OpenAI API (requires API key)", "OAI", ""],
    ["InferKit API (requires API key)", "InferKit", ""],
    ["KoboldAI Server API (Old Google Colab)", "Colab", ""],
    ["Return to Main Menu", "Return", ""],
]


# Variables
class Variables:
    lastact = ""  # The last action received from the user
    submission = ""  # Same as above, but after applying input formatting
    lastctx = ""  # The last context submitted to the generator
    model = ""  # Model ID string chosen at startup
    model_type = ""  # Model Type (Automatically taken from the model config)
    noai = False  # Runs the script without starting up the transformers pipeline
    aibusy = False  # Stops submissions while the AI is working
    max_length = 1024  # Maximum number of tokens to submit per action
    ikmax = 3000  # Maximum number of characters to submit to InferKit
    genamt = 80  # Amount of text for each action to generate
    ikgen = 200  # Number of characters for InferKit to generate
    rep_pen = 1.1  # Default generator repetition_penalty
    rep_pen_slope = 1.0  # Default generator repetition penalty slope
    rep_pen_range = 1024  # Default generator repetition penalty range
    temp = 0.5  # Default generator temperature
    top_p = 0.9  # Default generator top_p
    top_k = 0  # Default generator top_k
    tfs = 1.0  # Default generator tfs (tail-free sampling)
    numseqs = 1  # Number of sequences to ask the generator to create
    gamestarted = False  # Whether the game has started (disables UI elements)
    gamesaved = True  # Whether or not current game is saved
    serverstarted = False  # Whether or not the Flask server has started
    prompt = ""  # Prompt
    memory = ""  # Text submitted to memory field
    authornote = ""  # Text submitted to Author's Note field
    authornotetemplate = "[Author's note: <|>]"  # Author's note template
    setauthornotetemplate = authornotetemplate  # Saved author's note template in settings
    andepth = 3  # How far back in history to append author's note
    actions = structures.KoboldStoryRegister()  # Actions submitted by user and AI
    actions_metadata = {}  # List of dictonaries, one dictonary for every action that contains information about the action like alternative options.
    # Contains at least the same number of items as actions. Back action will remove an item from actions, but not actions_metadata
    # Dictonary keys are:
    # Selected Text: (text the user had selected. None when this is a newly generated action)
    # Alternative Generated Text: {Text, Pinned, Previous Selection, Edited}
    #
    worldinfo: list[Any] = []  # List of World Info key/value objects
    worldinfo_i = []  # List of World Info key/value objects sans uninitialized entries
    worldinfo_u = {}  # Dictionary of World Info UID - key/value pairs
    wifolders_d = {}  # Dictionary of World Info folder UID-info pairs
    wifolders_l = []  # List of World Info folder UIDs
    wifolders_u = {}  # Dictionary of pairs of folder UID - list of WI UID
    modelconfig = {}  # Raw contents of the model's config.json, or empty dictionary if none found
    lua_state = None  # Lua state of the Lua scripting system
    lua_koboldbridge = None  # `koboldbridge` from bridge.lua
    lua_kobold = None  # `kobold` from` bridge.lua
    lua_koboldcore = None  # `koboldcore` from bridge.lua
    lua_logname = ...  # Name of previous userscript that logged to terminal
    lua_running = False  # Whether or not Lua is running (i.e. wasn't stopped due to an error)
    lua_edited = set()  # Set of chunk numbers that were edited from a Lua generation modifier
    lua_deleted = set()  # Set of chunk numbers that were deleted from a Lua generation modifier
    generated_tkns = 0  # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
    abort = False  # Whether or not generation was aborted by clicking on the submit button during generation
    compiling = False  # If using a TPU Colab, this will be set to True when the TPU backend starts compiling and then set to False again
    checking = False  # Whether or not we are actively checking to see if TPU backend is compiling or not
    spfilename = ""  # Filename of soft prompt to load, or an empty string if not using a soft prompt
    userscripts = []  # List of userscripts to load
    last_userscripts = []  # List of previous userscript filenames from the previous time userscripts were send via usstatitems
    corescript = "default.lua"  # Filename of corescript to load
    # badwords    = []     # Array of str/chr values that should be removed from output
    badwordsids = [[13460], [6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922],
                   [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218],
                   [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406],
                   [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516],
                   [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279],
                   [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330],
                   [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841],
                   [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984],
                   [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250],
                   [3556], [38381], [36338], [32756], [46581], [17912],
                   [49146]]  # Tokenized array of badwords used to prevent AI artifacting
    deletewi = None  # Temporary storage for UID to delete
    wirmvwhtsp = False  # Whether to remove leading whitespace from WI entries
    widepth = 3  # How many historical actions to scan for WI hits
    mode = "play"  # Whether the interface is in play, memory, or edit mode
    editln = 0  # Which line was last selected in Edit Mode
    gpu_device = 0  # Which PyTorch device to use when using pure GPU generation
    url = "https://api.inferkit.com/v1/models/standard/generate"  # InferKit API URL
    oaiurl = ""  # OpenAI API URL
    oaiengines = "https://api.openai.com/v1/engines"
    colaburl = ""  # Ngrok url for Google Colab mode
    apikey = ""  # API key to use for InferKit API calls
    oaiapikey = ""  # API key to use for OpenAI API calls
    savedir = getcwd() + "\\stories"
    hascuda = False  # Whether torch has detected CUDA on the system
    usegpu = False  # Whether to launch pipeline with GPU support
    custmodpth = ""  # Filesystem location of custom model to run
    formatoptns = {'frmttriminc': True, 'frmtrmblln': False, 'frmtrmspch': False, 'frmtadsnsp': False,
                   'singleline': False}  # Container for state of formatting options
    importnum = -1  # Selection on import popup list
    importjs = {}  # Temporary storage for import data
    loadselect = ""  # Temporary storage for story filename to load
    spselect = ""  # Temporary storage for soft prompt filename to load
    spmeta = None  # Metadata of current soft prompt, or None if not using a soft prompt
    sp = None  # Current soft prompt tensor (as a NumPy array)
    sp_length = 0  # Length of current soft prompt in tokens, or 0 if not using a soft prompt
    has_genmod = False  # Whether or not at least one loaded Lua userscript has a generation modifier
    svowname = ""  # Filename that was flagged for overwrite confirm
    saveow = False  # Whether or not overwrite confirm has been displayed
    autosave = False  # Whether or not to automatically save after each action
    genseqs = []  # Temporary storage for generated sequences
    recentback = False  # Whether Back button was recently used without Submitting or Retrying after
    recentrng = None  # If a new random game was recently generated without Submitting after, this is the topic used (as a string), otherwise this is None
    recentrngm = None  # If a new random game was recently generated without Submitting after, this is the memory used (as a string), otherwise this is None
    useprompt = False  # Whether to send the full prompt with every submit action
    usebreakmodel = False  # For GPU users, whether to use both system RAM and VRAM to conserve VRAM while offering speedup compared to CPU-only
    bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J/XGLM only, currently)
    nobreakmodel = False  # Something specifically requested Breakmodel to be disabled (For example a models config)
    smandelete = False  # Whether stories can be deleted from inside the browser
    smanrename = False  # Whether stories can be renamed from inside the browser
    allowsp = False  # Whether we are allowed to use soft prompts (by default enabled if we're using GPT-2, GPT-Neo or GPT-J)
    modeldim = -1  # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
    laststory = None  # Filename (without extension) of most recent story JSON file we loaded
    regex_sl = re.compile(r'\n*(?<=.) *\n(.|\n)*')  # Pattern for limiting the output to a single line
    acregex_ai = re.compile(
        r'\n* *>(.|\n)*')  # Pattern for matching adventure actions from the AI so we can remove them
    acregex_ui = re.compile(r'^ *(&gt;.*)$',
                            re.MULTILINE)  # Pattern for matching actions in the HTML-escaped story so we can apply colouring, etc (make sure to encase part to format in parentheses)
    comregex_ai = re.compile(
        r'(?:\n<\|(?:.|\n)*?\|>(?=\n|$))|(?:<\|(?:.|\n)*?\|>\n?)')  # Pattern for matching comments to remove them before sending them to the AI
    comregex_ui = re.compile(r'(&lt;\|(?:.|\n)*?\|&gt;)')  # Pattern for matching comments in the editor
    chatmode = False
    chatname = "You"
    adventure = False
    actionmode = 1
    dynamicscan = False
    host = False
    nopromptgen = False
    rngpersist = False
    nogenmod = False
    welcome = False  # Custom Welcome Text (False is default)
    newlinemode = "n"
    quiet = False  # If set will suppress any story text from being printed to the console (will only be seen on the client web page)
    debug = False  # If set to true, will send debug information to the client for display
    lazy_load = True  # Whether or not to use torch_lazy_loader.py for transformers models in order to reduce CPU memory usage
    use_colab_tpu = os.environ.get("COLAB_TPU_ADDR",
                                   "") != ""  # Whether or not we're in a Colab TPU instance and are going to use the TPU rather than the CPU


utils.vars = vars


# ==================================================================#
# Function to get model selection at startup
# ==================================================================#
def getmodelselection(modellist):
    print("    #    Model\t\t\t\t\t\tVRAM\n    ========================================================")
    i = 1
    for m in modellist:
        print("    {0} - {1}\t\t\t{2}".format("{:<2}".format(i), m[0].ljust(25), m[2]))
        i += 1
    print(" ")
    Variables.model = ''
    while Variables.model == '':
        modelsel = input("Model #> ")
        if modelsel.isnumeric() and 0 < int(modelsel) <= len(modellist):
            Variables.model = modellist[int(modelsel) - 1][1]
        else:
            print("{0}Please enter a valid selection.{1}".format(Colors.RED, Colors.END))

    # Model Lists
    try:
        getmodelselection(eval(Variables.model))
    except Exception:
        if Variables.model == "Return":
            getmodelselection(mainmenu)

        # If custom model was selected, get the filesystem location and store it
        if Variables.model == "NeoCustom" or Variables.model == "GPT2Custom":
            print(
                "{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(Colors.CYAN, Colors.END))
            modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")

            if modpath:
                # Save directory to vars
                Variables.custmodpth = modpath
            else:
                # Print error and retry model selection
                print("{0}Model select cancelled!{1}".format(Colors.RED, Colors.END))
                print("{0}Select an AI model to continue:{1}\n".format(Colors.CYAN, Colors.END))
                getmodelselection(mainmenu)


# ==================================================================
# Return Model Name
# ==================================================================
def getmodelname():
    if args.configname:
        modelname = args.configname
        return modelname
    if Variables.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ"):
        modelname = os.path.basename(os.path.normpath(Variables.custmodpth))
        return modelname
    else:
        modelname = Variables.model
        return modelname


# ==================================================================#
# Breakmodel configuration functions
# ==================================================================#
def device_list(n_layers, primary=None, selected=None):
    device_count = torch.cuda.device_count()
    if device_count < 2:
        primary = None
    gpu_blocks = breakmodel.gpu_blocks + (device_count - len(breakmodel.gpu_blocks)) * [0]
    print(f"{Colors.YELLOW}       DEVICE ID  |  LAYERS  |  DEVICE NAME{Colors.END}")
    for device_number in range(device_count):
        name = torch.cuda.get_device_name(device_number)
        if len(name) > 47:
            name = "..." + name[-44:]
        row_color = Colors.END
        sep_color = Colors.YELLOW
        print(
            f"{row_color}{Colors.YELLOW + '->' + row_color if device_number == selected else '  '} "
            f"{'(primary)' if device_number == primary else ' ' * 9} {device_number:3}  {sep_color}|{row_color}"
            f"     {gpu_blocks[device_number]:3}  {sep_color}|{row_color}  {name}{Colors.END}")
    row_color = Colors.END
    sep_color = Colors.YELLOW
    print(
        f"{row_color}   {' ' * 9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}"
        f"  (CPU){Colors.END}")


def device_config(config):
    global breakmodel, generator
    import breakmodel

    n_layers = config.num_layers if hasattr(config, "num_layers") else config.n_layer
    if args.breakmodel_gpulayers is not None:
        try:
            breakmodel.gpu_blocks = list(map(int, args.breakmodel_gpulayers.split(',')))
            assert len(breakmodel.gpu_blocks) <= torch.cuda.device_count()
            s = n_layers
            for block in range(len(breakmodel.gpu_blocks)):
                if breakmodel.gpu_blocks[block] <= -1:
                    breakmodel.gpu_blocks[block] = s
                    break
                else:
                    s -= breakmodel.gpu_blocks[block]
            assert sum(breakmodel.gpu_blocks) <= n_layers
            n_layers -= sum(breakmodel.gpu_blocks)
        except:
            print(
                "WARNING: --breakmodel_gpulayers is malformatted. Please use the --help option to see correct usage "
                "of --breakmodel_gpulayers. Defaulting to all layers on device 0.",
                file=sys.stderr)
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
    elif args.breakmodel_layers is not None:
        breakmodel.gpu_blocks = [n_layers - max(0, min(n_layers, args.breakmodel_layers))]
        n_layers -= sum(breakmodel.gpu_blocks)
    elif args.model is not None:
        print("Breakmodel not specified, assuming GPU 0")
        breakmodel.gpu_blocks = [n_layers]
        n_layers = 0
    else:
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(Colors.CYAN + "\nPlease select one of your GPUs to be your primary GPU.")
            print("VRAM usage in your primary GPU will be higher than for your other ones.")
            print("It is recommended you make your fastest GPU your primary GPU.")
            device_list(n_layers)
            while True:
                primaryselect = input("device ID> ")
                if primaryselect.isnumeric() and 0 <= int(primaryselect) < device_count:
                    breakmodel.primary_device = int(primaryselect)
                    break
                else:
                    print(f"{Colors.RED}Please enter an integer between 0 and {device_count - 1}.{Colors.END}")
        else:
            breakmodel.primary_device = 0

        print(Colors.PURPLE + "\nIf you don't have enough VRAM to run the model on a single GPU")
        print("you can split the model between your CPU and your GPU(s), or between")
        print("multiple GPUs if you have more than one.")
        print("By putting more 'layers' on a GPU or CPU, more computations will be")
        print("done on that device and more VRAM or RAM will be required on that device")
        print("(roughly proportional to number of layers).")
        print("It should be noted that GPUs are orders of magnitude faster than the CPU.")
        print(f"This model has{Colors.YELLOW} {n_layers} {Colors.PURPLE}layers.{Colors.END}\n")

        for i in range(device_count):
            device_list(n_layers, primary=breakmodel.primary_device, selected=i)
            print(
                f"{Colors.CYAN}\nHow many of the remaining{Colors.YELLOW} {n_layers} {Colors.CYAN}"
                f"layers would you like to put into device {i}?\nYou can also enter -1 to allocate all remaining layers"
                f" to this device.{Colors.END}\n")
            while True:
                layerselect = input("# of layers> ")
                if (layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers:
                    layerselect = int(layerselect)
                    layerselect = n_layers if layerselect == -1 else layerselect
                    breakmodel.gpu_blocks.append(layerselect)
                    n_layers -= layerselect
                    break
                else:
                    print(f"{Colors.RED}Please enter an integer between -1 and {n_layers}.{Colors.END}")
            if n_layers == 0:
                break

    print(Colors.PURPLE + "\nFinal device configuration:")
    device_list(n_layers)

    # If all layers are on the same device, use the old GPU generation mode
    while len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0:
        breakmodel.gpu_blocks.pop()
    if (len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (
            -1, config.num_layers if hasattr(config, "num_layers") else config.n_layer)):
        Variables.breakmodel = False
        Variables.usegpu = True
        Variables.gpu_device = len(breakmodel.gpu_blocks) - 1
        return

    if not breakmodel.gpu_blocks:
        print("Nothing assigned to a GPU, reverting to CPU only mode")
        Variables.breakmodel = False
        Variables.usegpu = False
        return


def move_model_to_devices(model):
    global generator

    if not Variables.usebreakmodel:
        if Variables.usegpu:
            model = model.half().to(Variables.gpu_device)
        else:
            model = model.to('cpu').float()
        generator = model.generate
        return

    model.half()
    gc.collect()
    if hasattr(model, "transformer"):
        model.transformer.wte.to(breakmodel.primary_device)
        model.transformer.ln_f.to(breakmodel.primary_device)
        if hasattr(model, 'lm_head'):
            model.lm_head.to(breakmodel.primary_device)
        if hasattr(model.transformer, 'wpe'):
            model.transformer.wpe.to(breakmodel.primary_device)
    else:
        model.model.embed_tokens.to(breakmodel.primary_device)
        model.model.layer_norm.to(breakmodel.primary_device)
        model.lm_head.to(breakmodel.primary_device)
        model.model.embed_positions.to(breakmodel.primary_device)
    gc.collect()
    GPTNeoModel.forward = breakmodel.new_forward_neo
    if "GPTJModel" in globals():
        GPTJModel.forward = breakmodel.new_forward_neo
    if "XGLMModel" in globals():
        XGLMModel.forward = breakmodel.new_forward_xglm
    generator = model.generate
    if hasattr(model, "transformer"):
        breakmodel.move_hidden_layers(model.transformer)
    else:
        breakmodel.move_hidden_layers(model.model, model.model.layers)


# ==================================================================#
#  Allow the models to override some settings
# ==================================================================#
def loadmodelsettings():
    try:
        js = json.loads(str(model_config).partition(' ')[2])
    except Exception:
        try:
            try:
                js = json.load(open(Variables.custmodpth + "/config.json", "r"))
            except Exception:
                js = json.load(open(Variables.custmodpth.replace('/', '_') + "/config.json", "r"))
        except Exception:
            js = {}
    if Variables.model_type == "xglm" or js.get("compat", "j") == "fairseq_lm":
        Variables.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    Variables.modelconfig = js
    if "badwordsids" in js:
        Variables.badwordsids = js["badwordsids"]
    if "nobreakmodel" in js:
        Variables.nobreakmodel = js["nobreakmodel"]
    if "temp" in js:
        Variables.temp = js["temp"]
    if "top_p" in js:
        Variables.top_p = js["top_p"]
    if "top_k" in js:
        Variables.top_k = js["top_k"]
    if "tfs" in js:
        Variables.tfs = js["tfs"]
    if "rep_pen" in js:
        Variables.rep_pen = js["rep_pen"]
    if "rep_pen_slope" in js:
        Variables.rep_pen_slope = js["rep_pen_slope"]
    if "rep_pen_range" in js:
        Variables.rep_pen_range = js["rep_pen_range"]
    if "adventure" in js:
        Variables.adventure = js["adventure"]
    if "chatmode" in js:
        Variables.chatmode = js["chatmode"]
    if "dynamicscan" in js:
        Variables.dynamicscan = js["dynamicscan"]
    if "formatoptns" in js:
        Variables.formatoptns = js["formatoptns"]
    if "welcome" in js:
        Variables.welcome = js["welcome"]
    if "newlinemode" in js:
        Variables.newlinemode = js["newlinemode"]
    if "antemplate" in js:
        Variables.setauthornotetemplate = js["antemplate"]
        if not Variables.gamestarted:
            Variables.authornotetemplate = Variables.setauthornotetemplate


# ==================================================================#
#  Take settings from vars and write them to client settings file
# ==================================================================#
def savesettings():
    # Build json to write
    js = {}
    js["apikey"] = Variables.apikey
    js["andepth"] = Variables.andepth
    js["temp"] = Variables.temp
    js["top_p"] = Variables.top_p
    js["top_k"] = Variables.top_k
    js["tfs"] = Variables.tfs
    js["rep_pen"] = Variables.rep_pen
    js["rep_pen_slope"] = Variables.rep_pen_slope
    js["rep_pen_range"] = Variables.rep_pen_range
    js["genamt"] = Variables.genamt
    js["max_length"] = Variables.max_length
    js["ikgen"] = Variables.ikgen
    js["formatoptns"] = Variables.formatoptns
    js["numseqs"] = Variables.numseqs
    js["widepth"] = Variables.widepth
    js["useprompt"] = Variables.useprompt
    js["adventure"] = Variables.adventure
    js["chatmode"] = Variables.chatmode
    js["chatname"] = Variables.chatname
    js["dynamicscan"] = Variables.dynamicscan
    js["nopromptgen"] = Variables.nopromptgen
    js["rngpersist"] = Variables.rngpersist
    js["nogenmod"] = Variables.nogenmod
    js["autosave"] = Variables.autosave
    js["welcome"] = Variables.welcome
    js["newlinemode"] = Variables.newlinemode

    js["antemplate"] = Variables.setauthornotetemplate

    js["userscripts"] = Variables.userscripts
    js["corescript"] = Variables.corescript
    js["softprompt"] = Variables.spfilename

    # Write it
    if not os.path.exists('settings'):
        os.mkdir('settings')
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
    try:
        file.write(json.dumps(js, indent=3))
    finally:
        file.close()


# ==================================================================#
#  Don't save settings unless 2 seconds have passed without modification
# ==================================================================#
@debounce(2)
def settingschanged():
    print("{0}Saving settings!{1}".format(Colors.GREEN, Colors.END))
    savesettings()


# ==================================================================#
#  Read settings from client file JSON and send to vars
# ==================================================================#
def loadsettings():
    if path.exists("settings/" + getmodelname().replace('/', '_') + ".settings"):
        # Read file contents into JSON object
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js = json.load(file)

        # Copy file contents to vars
        if "apikey" in js:
            Variables.apikey = js["apikey"]
        if "andepth" in js:
            Variables.andepth = js["andepth"]
        if "temp" in js:
            Variables.temp = js["temp"]
        if "top_p" in js:
            Variables.top_p = js["top_p"]
        if "top_k" in js:
            Variables.top_k = js["top_k"]
        if "tfs" in js:
            Variables.tfs = js["tfs"]
        if "rep_pen" in js:
            Variables.rep_pen = js["rep_pen"]
        if "rep_pen_slope" in js:
            Variables.rep_pen_slope = js["rep_pen_slope"]
        if "rep_pen_range" in js:
            Variables.rep_pen_range = js["rep_pen_range"]
        if "genamt" in js:
            Variables.genamt = js["genamt"]
        if "max_length" in js:
            Variables.max_length = js["max_length"]
        if "ikgen" in js:
            Variables.ikgen = js["ikgen"]
        if "formatoptns" in js:
            Variables.formatoptns = js["formatoptns"]
        if "numseqs" in js:
            Variables.numseqs = js["numseqs"]
        if "widepth" in js:
            Variables.widepth = js["widepth"]
        if "useprompt" in js:
            Variables.useprompt = js["useprompt"]
        if "adventure" in js:
            Variables.adventure = js["adventure"]
        if "chatmode" in js:
            Variables.chatmode = js["chatmode"]
        if "chatname" in js:
            Variables.chatname = js["chatname"]
        if "dynamicscan" in js:
            Variables.dynamicscan = js["dynamicscan"]
        if "nopromptgen" in js:
            Variables.nopromptgen = js["nopromptgen"]
        if "rngpersist" in js:
            Variables.rngpersist = js["rngpersist"]
        if "nogenmod" in js:
            Variables.nogenmod = js["nogenmod"]
        if "autosave" in js:
            Variables.autosave = js["autosave"]
        if "newlinemode" in js:
            Variables.newlinemode = js["newlinemode"]
        if "welcome" in js:
            Variables.welcome = js["welcome"]

        if "antemplate" in js:
            Variables.setauthornotetemplate = js["antemplate"]
            if not Variables.gamestarted:
                Variables.authornotetemplate = Variables.setauthornotetemplate

        if "userscripts" in js:
            Variables.userscripts = []
            for userscript in js["userscripts"]:
                if type(userscript) is not str:
                    continue
                userscript = userscript.strip()
                if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(
                        userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                    Variables.userscripts.append(userscript)

        if "corescript" in js and type(js["corescript"]) is str and all(
                q not in js["corescript"] for q in ("..", ":")) and all(
                js["corescript"][0] not in q for q in ("/", "\\")):
            Variables.corescript = js["corescript"]
        else:
            Variables.corescript = "default.lua"

        file.close()


# ==================================================================#
#  Load a soft prompt from a file
# ==================================================================#
def sprequest(filename):
    global np
    Variables.spfilename = ""
    settingschanged()

    if len(filename) == 0:
        Variables.sp = None
        Variables.sp_length = 0
        return

    z, version, shape, fortran_order, dtype = fileops.checksp(filename, Variables.modeldim)
    assert isinstance(z, zipfile.ZipFile)
    with z.open('meta.json') as f:
        Variables.spmeta = json.load(f)
    z.close()

    with np.load(fileops.sppath(filename), allow_pickle=False) as f:
        tensor = f['tensor.npy']

    # If the tensor is in bfloat16 format, convert it to float32
    if tensor.dtype == 'V2':
        tensor.dtype = np.uint16
        tensor = np.uint32(tensor) << 16
        tensor.dtype = np.float32

    if tensor.dtype != np.float16:
        tensor = np.float32(tensor)
    assert not np.isinf(tensor).any() and not np.isnan(tensor).any()

    Variables.sp_length = tensor.shape[-2]
    Variables.spmeta["n_tokens"] = Variables.sp_length

    if Variables.use_colab_tpu or Variables.model in ("TPUMeshTransformerGPTJ",):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (
                tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params["d_model"],
        )
        Variables.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        Variables.sp = torch.from_numpy(tensor)

    Variables.spfilename = filename
    settingschanged()


# ==================================================================#
# Startup
# ==================================================================#

# Parsing Parameters
parser = argparse.ArgumentParser(description="KoboldAI Server")
parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
parser.add_argument("--ngrok", action='store_true', help="Optimizes KoboldAI for Remote Play using Ngrok")
parser.add_argument("--host", action='store_true',
                    help="Optimizes KoboldAI for Remote Play without using a proxy service")
parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
parser.add_argument("--cpu", action='store_true',
                    help="By default unattended launches are on the GPU use this option to force CPU usage.")
parser.add_argument("--breakmodel", action='store_true', help=argparse.SUPPRESS)
parser.add_argument("--breakmodel_layers", type=int, help=argparse.SUPPRESS)
parser.add_argument("--breakmodel_gpulayers", type=str,
                    help="If using a model that supports hybrid generation, this is a comma-separated list that specifies how many layers to put on each GPU device. For example to put 8 layers on device 0, 9 layers on device 1 and 11 layers on device 2, use --beakmodel_gpulayers 8,9,11")
parser.add_argument("--override_delete", action='store_true',
                    help="Deleting stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow deleting stories if using --remote and prevent deleting stories otherwise.")
parser.add_argument("--override_rename", action='store_true',
                    help="Renaming stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow renaming stories if using --remote and prevent renaming stories otherwise.")
parser.add_argument("--configname", help="Force a fixed configuration name to aid with config management.")
parser.add_argument("--colab", action='store_true', help="Optimize for Google Colab.")
parser.add_argument("--nobreakmodel", action='store_true', help="Disables Breakmodel support completely.")
parser.add_argument("--unblock", action='store_true', default=False,
                    help="Unblocks the KoboldAI port to be accessible from other machines without optimizing for remote play (It is recommended to use --host instead)")
parser.add_argument("--quiet", action='store_true', default=False,
                    help="If present will suppress any story related text from showing on the console")
parser.add_argument("--lowmem", action='store_true',
                    help="Extra Low Memory loading for the GPU, slower but memory does not peak to twice the usage")

args: argparse.Namespace = None
if os.environ.get("KOBOLDAI_ARGS") is not None:
    import shlex

    args = parser.parse_args(shlex.split(os.environ["KOBOLDAI_ARGS"]))
else:
    args = parser.parse_args()

Variables.model = args.model

if args.colab:
    args.remote = True
    args.override_rename = True
    args.override_delete = True
    args.nobreakmodel = True
    args.quiet = True
    args.lowmem = True

if args.quiet:
    Variables.quiet = True

if args.nobreakmodel:
    Variables.nobreakmodel = True

if args.remote:
    Variables.host = True

if args.ngrok:
    Variables.host = True

if args.host:
    Variables.host = True

if args.cpu:
    Variables.use_colab_tpu = False

Variables.smandelete = Variables.host == args.override_delete
Variables.smanrename = Variables.host == args.override_rename

# Select a model to run
if args.model:
    print("Welcome to KoboldAI!\nYou have selected the following Model:", Variables.model)
    if args.path:
        print("You have selected the following path for your Model :", args.path)
        Variables.custmodpth = args.path
        Variables.colaburl = args.path + "/request"  # Lets just use the same parameter to keep it simple

else:
    print(
        "{0}Welcome to the KoboldAI Server!\nListed RAM is the optimal VRAM and CPU ram can be up to twice the "
        "amount.\nMost models can run at less VRAM with reduced max tokens or less layers on the GPU.\nSelect an AI "
        "model to continue:{1}\n".format(
            Colors.CYAN, Colors.END))
    getmodelselection(mainmenu)

# If transformers model was selected & GPU available, ask to use CPU or GPU
if Variables.model not in ["InferKit", "Colab", "OAI", "GooseAI", "ReadOnly", "TPUMeshTransformerGPTJ"]:
    Variables.allowsp = True
    # Test for GPU support
    import torch

    # Make model path the same as the model name to make this consistent with the other loading method if it isn't a
    # known model type This code is not just a workaround for below, it is also used to make the behavior consistent
    # with other loading methods - Henk717
    if Variables.model not in ["NeoCustom", "GPT2Custom"]:
        Variables.custmodpth = Variables.model
    elif Variables.model == "NeoCustom":
        Variables.model = os.path.basename(os.path.normpath(Variables.custmodpth))

    # Get the model_type from the config or assume a model type if it isn't present
    from transformers import AutoConfig, GPTJModel, XGLMModel

    if os.path.isdir(Variables.custmodpth.replace('/', '_')):
        try:
            model_config = AutoConfig.from_pretrained(Variables.custmodpth.replace('/', '_'), cache_dir="cache/")
            Variables.model_type = model_config.model_type
        except ValueError as e:
            Variables.model_type = "not_found"
    elif os.path.isdir("models/{}".format(Variables.custmodpth.replace('/', '_'))):
        try:
            model_config = AutoConfig.from_pretrained("models/{}".format(Variables.custmodpth.replace('/', '_')),
                                                      cache_dir="cache/")
            Variables.model_type = model_config.model_type
        except ValueError as e:
            Variables.model_type = "not_found"
    else:
        try:
            model_config = AutoConfig.from_pretrained(Variables.custmodpth, cache_dir="cache/")
            Variables.model_type = model_config.model_type
        except ValueError as e:
            Variables.model_type = "not_found"
    if Variables.model_type == "not_found" and Variables.model == "NeoCustom":
        Variables.model_type = "gpt_neo"
    elif Variables.model_type == "not_found" and Variables.model == "GPT2Custom":
        Variables.model_type = "gpt2"
    elif Variables.model_type == "not_found":
        print(
            "WARNING: No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or "
            "--model GPT2Custom)")
        Variables.model_type = "gpt_neo"

if (not Variables.use_colab_tpu and Variables.model not in ["InferKit", "Colab", "OAI", "GooseAI", "ReadOnly",
                                                            "TPUMeshTransformerGPTJ"]):
    loadmodelsettings()
    loadsettings()
    print("{0}Looking for GPU support...{1}".format(Colors.PURPLE, Colors.END), end="")
    Variables.hascuda = torch.cuda.is_available()
    Variables.bmsupported = Variables.model_type in ("gpt_neo", "gptj", "xglm") and not Variables.nobreakmodel
    if args.breakmodel is not None and args.breakmodel:
        print(
            "WARNING: --breakmodel is no longer supported. Breakmodel mode is now automatically enabled when "
            "--breakmodel_gpulayers is used (see --help for details).",
            file=sys.stderr)
    if args.breakmodel_layers is not None:
        print(
            "WARNING: --breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details).",
            file=sys.stderr)
    if args.model and Variables.bmsupported and not args.breakmodel_gpulayers and not args.breakmodel_layers:
        print("WARNING: Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode.",
              file=sys.stderr)
        Variables.bmsupported = False
    if not Variables.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None):
        print("WARNING: This model does not support hybrid generation. --breakmodel_gpulayers will be ignored.",
              file=sys.stderr)
    if Variables.hascuda:
        print("{0}FOUND!{1}".format(Colors.GREEN, Colors.END))
    else:
        print("{0}NOT FOUND!{1}".format(Colors.YELLOW, Colors.END))

    genselected = False
    if args.model:
        if Variables.hascuda:
            genselected = True
            Variables.usegpu = True
            Variables.breakmodel = False
        if Variables.bmsupported:
            Variables.usegpu = False
            Variables.breakmodel = True
        if args.cpu:
            Variables.usegpu = False
            Variables.breakmodel = False
    elif Variables.hascuda:
        if Variables.bmsupported:
            genselected = True
            Variables.usegpu = False
            Variables.breakmodel = True
        else:
            print("    1 - GPU\n    2 - CPU\n")
            genselected = False

    if Variables.hascuda:
        while not genselected:
            genselect = input("Mode> ")
            if genselect == "":
                Variables.breakmodel = False
                Variables.usegpu = True
                genselected = True
            elif genselect.isnumeric() and int(genselect) == 1:
                if Variables.bmsupported:
                    Variables.breakmodel = True
                    Variables.usegpu = False
                    genselected = True
                else:
                    Variables.breakmodel = False
                    Variables.usegpu = True
                    genselected = True
            elif genselect.isnumeric() and int(genselect) == 2:
                Variables.breakmodel = False
                Variables.usegpu = False
                genselected = True
            else:
                print("{0}Please enter a valid selection.{1}".format(Colors.RED, Colors.END))

# Ask for API key if InferKit was selected
if Variables.model == "InferKit":
    if not path.exists("settings/" + getmodelname().replace('/', '_') + ".settings"):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your InferKit API key:{1}\n".format(Colors.CYAN, Colors.END))
        Variables.apikey = input("Key> ")
        # Write API key to file
        os.makedirs('settings', exist_ok=True)
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
        try:
            js = {"apikey": Variables.apikey}
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
    else:
        # Otherwise open it up
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        # Check if API key exists
        js = json.load(file)
        if "apikey" in js and js["apikey"] != "":
            # API key exists, grab it and close the file
            Variables.apikey = js["apikey"]
            file.close()
        else:
            # Get API key, add it to settings object, and write it to disk
            print("{0}Please enter your InferKit API key:{1}\n".format(Colors.CYAN, Colors.END))
            Variables.apikey = input("Key> ")
            js["apikey"] = Variables.apikey
            # Write API key to file
            file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
            try:
                file.write(json.dumps(js, indent=3))
            finally:
                file.close()

# Swap OAI Server if GooseAI was selected
if Variables.model == "GooseAI":
    Variables.oaiengines = "https://api.goose.ai/v1/engines"
    Variables.model = "OAI"
    args.configname = "GooseAI"

# Ask for API key if OpenAI was selected
if Variables.model == "OAI":
    if not args.configname:
        args.configname = "OAI"
    if not path.exists("settings/" + getmodelname().replace('/', '_') + ".settings"):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your API key:{1}\n".format(Colors.CYAN, Colors.END))
        Variables.oaiapikey = input("Key> ")
        # Write API key to file
        os.makedirs('settings', exist_ok=True)
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
        try:
            js = {"oaiapikey": Variables.oaiapikey}
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
    else:
        # Otherwise open it up
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        # Check if API key exists
        js = json.load(file)
        if "oaiapikey" in js and js["oaiapikey"] != "":
            # API key exists, grab it and close the file
            Variables.oaiapikey = js["oaiapikey"]
            file.close()
        else:
            # Get API key, add it to settings object, and write it to disk
            print("{0}Please enter your API key:{1}\n".format(Colors.CYAN, Colors.END))
            Variables.oaiapikey = input("Key> ")
            js["oaiapikey"] = Variables.oaiapikey
            # Write API key to file
            file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
            try:
                file.write(json.dumps(js, indent=3))
            finally:
                file.close()

    # Get list of models from OAI
    print("{0}Retrieving engine list...{1}".format(Colors.PURPLE, Colors.END), end="")
    req = requests.get(
        Variables.oaiengines,
        headers={
            'Authorization': 'Bearer ' + Variables.oaiapikey
        }
    )
    if req.status_code == 200:
        print("{0}OK!{1}".format(Colors.GREEN, Colors.END))
        print("{0}Please select an engine to use:{1}\n".format(Colors.CYAN, Colors.END))
        engines = req.json()["data"]
        # Print list of engines
        i = 0
        for en in engines:
            print("    {0} - {1} ({2})".format(i, en["id"],
                                               "\033[92mready\033[0m" if en["ready"] is True
                                               else "\033[""91mnot ready\033[0m"))
            i += 1
        # Get engine to use
        print("")
        engselected = False
        while not engselected:
            engine = input("Engine #> ")
            if engine.isnumeric() and int(engine) < len(engines):
                Variables.oaiurl = Variables.oaiengines + "/{0}/completions".format(engines[int(engine)]["id"])
                args.configname = args.configname + "/" + engines[int(engine)]["id"]
                engselected = True
            else:
                print("{0}Please enter a valid selection.{1}".format(Colors.RED, Colors.END))
    else:
        # Something went wrong, print the message and quit since we can't initialize an engine
        print("{0}ERROR!{1}".format(Colors.RED, Colors.END))
        print(req.json())
        quit()

# Ask for ngrok url if Google Colab was selected
if Variables.model == "Colab":
    if Variables.colaburl == "":
        print(
            "{0}NOTE: For the modern KoboldAI Colab's you open the links directly in your browser.\nThis option is "
            "only for the KoboldAI Server API, not all features are supported in this mode.\n".format(
                Colors.YELLOW, Colors.END))
        print("{0}Enter the URL of the server (For example a trycloudflare link):{1}\n".format(Colors.CYAN, Colors.END))
        Variables.colaburl = input("URL> ") + "/request"

if Variables.model == "ReadOnly":
    Variables.noai = True

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(Colors.PURPLE, Colors.END), end="")

app = Flask(__name__)
app.config['SECRET KEY'] = 'secret!'
socketio = SocketIO(app, async_method="eventlet")
print("{0}OK!{1}".format(Colors.GREEN, Colors.END))

# Start transformers and create pipeline
if (not Variables.use_colab_tpu and Variables.model not in ["InferKit", "Colab", "OAI", "GooseAI", "ReadOnly",
                                                            "TPUMeshTransformerGPTJ"]):
    if not Variables.noai:
        print("{0}Initializing transformers, please wait...{1}".format(Colors.PURPLE, Colors.END))
        from transformers import StoppingCriteria, GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, \
            AutoModelForCausalLM, AutoTokenizer

        for m in ("GPTJModel", "XGLMModel"):
            try:
                globals()[m] = getattr(__import__("transformers"), m)
            except:
                pass
        import transformers.generation_utils
        from transformers import __version__ as transformers_version

        # Lazy loader
        import torch_lazy_loader


        def get_lazy_load_callback(n_layers, convert_to_float16=True):
            if not Variables.lazy_load:
                return

            from tqdm import tqdm

            if "breakmodel" in globals():
                gpu_blocks = breakmodel.gpu_blocks
                ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
                cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))
            else:
                ram_blocks = None
                cumulative_gpu_blocks = None

            def lazy_load_callback(model_dict, f, **_):
                device_map = {}

                for _key, spec in lazy_load_spec.get("layer_weights", {}).items():
                    for layer in range(n_layers):
                        key = _key.format(layer=layer)
                        if key not in model_dict:
                            continue
                        device = Variables.gpu_device if Variables.hascuda and Variables.usegpu else "cpu" if not Variables.hascuda or not Variables.breakmodel or layer < ram_blocks else bisect.bisect_right(
                            cumulative_gpu_blocks, layer - ram_blocks)
                        device_map[key] = device

                for key, value in model_dict.items():
                    if isinstance(value, torch_lazy_loader.LazyTensor) and key not in device_map:
                        device_map[key] = Variables.gpu_device if Variables.hascuda and Variables.usegpu else "cpu"

                with zipfile.ZipFile(f, "r") as z:
                    try:
                        last_storage_key = None
                        f = None
                        for key in tqdm(
                                sorted(device_map.keys(), key=lambda k: (model_dict[k].key, model_dict[k].seek_offset)),
                                desc="Loading model tensors"):
                            storage_key = model_dict[key].key
                            if storage_key != last_storage_key:
                                last_storage_key = storage_key
                                if isinstance(f, zipfile.ZipExtFile):
                                    f.close()
                                f = z.open(f"archive/data/{storage_key}")
                            current_offset = f.tell()
                            if current_offset != model_dict[key].seek_offset:
                                f.seek(model_dict[key].seek_offset - current_offset, 1)
                            device = device_map[key]
                            # print(f"Transferring <{key}>  to  {'(CPU)' if device == 'cpu' else '[device ' + str(device) + ']'} ... ", end="", flush=True)
                            model_dict[key] = model_dict[key].materialize(f, map_location="cpu")
                            if convert_to_float16 and Variables.hascuda and (
                                    Variables.breakmodel or Variables.usegpu) and \
                                    model_dict[key].dtype is torch.float32:
                                model_dict[key] = model_dict[key].to(torch.float16)
                            if not Variables.usegpu and not Variables.breakmodel and \
                                    model_dict[key].dtype is torch.float16:
                                model_dict[key] = model_dict[key].to(torch.float32)
                            model_dict[key] = model_dict[key].to(device)
                            # print("OK", flush=True)
                    finally:
                        if isinstance(f, zipfile.ZipExtFile):
                            f.close()

            return lazy_load_callback


        lazy_load_config_path = os.path.join(path.dirname(path.realpath(__file__)), "maps",
                                             Variables.model_type + ".json")
        if Variables.lazy_load and "model_config" in globals() and os.path.isfile(lazy_load_config_path):
            with open(lazy_load_config_path) as f:
                lazy_load_spec = json.load(f)

        else:
            Variables.lazy_load = False


        # Patch transformers to use our soft prompt
        def patch_causallm(cls):
            old_forward = cls.forward

            def new_causallm_forward(self, *args, **kwargs):
                input_ids = kwargs.get('input_ids').to(self.device)
                shifted_input_ids = 0
                assert input_ids is not None
                kwargs['input_ids'] = None
                if Variables.sp is not None:
                    shifted_input_ids = input_ids - self.config.vocab_size
                input_ids.clamp_(max=self.config.vocab_size - 1)
                if hasattr(self, "transformer"):
                    inputs_embeds = self.transformer.wte(input_ids)
                else:
                    inputs_embeds = self.model.embed_tokens(input_ids)
                if Variables.sp is not None:
                    Variables.sp = Variables.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
                    inputs_embeds = torch.where(
                        (shifted_input_ids >= 0)[..., None],
                        Variables.sp[shifted_input_ids.clamp(min=0)],
                        inputs_embeds,
                    )
                if not hasattr(self, "transformer"):
                    inputs_embeds *= self.model.embed_scale
                kwargs['inputs_embeds'] = inputs_embeds
                return old_forward(self, *args, **kwargs)

            cls.forward = new_causallm_forward


        for cls in (GPT2LMHeadModel, GPTNeoForCausalLM):
            patch_causallm(cls)
        for c in ("GPTJForCausalLM", "XGLMForCausalLM"):
            try:
                patch_causallm(getattr(__import__("transformers"), c))
            except ImportError:
                pass

        # Patch transformers to use our custom logit warpers
        from transformers import LogitsProcessorList, LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, \
            TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
        from warpers import AdvancedRepetitionPenaltyLogitsProcessor, TailFreeLogitsWarper


        def dynamic_processor_wrap(cls, field_name, var_name, cond=None):
            old_call = cls.__call__

            def new_call(self, *args, **kwargs):
                if not isinstance(field_name, str) and isinstance(field_name, Iterable):
                    conds = []
                    for f, v in zip(field_name, var_name):
                        conds.append(getattr(vars, v))
                        setattr(self, f, conds[-1])
                else:
                    conds = getattr(vars, var_name)
                    setattr(self, field_name, conds)
                assert len(args) == 2
                if cond is None or cond(conds):
                    return old_call(self, *args, **kwargs)
                return args[1]

            cls.__call__ = new_call


        dynamic_processor_wrap(AdvancedRepetitionPenaltyLogitsProcessor, ("penalty", "penalty_slope", "penalty_range"),
                               ("rep_pen", "rep_pen_slope", "rep_pen_range"), cond=lambda x: x[0] != 1.0)
        dynamic_processor_wrap(TopKLogitsWarper, "top_k", "top_k", cond=lambda x: x > 0)
        dynamic_processor_wrap(TopPLogitsWarper, "top_p", "top_p", cond=lambda x: x < 1.0)
        dynamic_processor_wrap(TailFreeLogitsWarper, "tfs", "tfs", cond=lambda x: x < 1.0)
        dynamic_processor_wrap(TemperatureLogitsWarper, "temperature", "temp", cond=lambda x: x != 1.0)
        RepetitionPenaltyLogitsProcessor.__init__ = AdvancedRepetitionPenaltyLogitsProcessor.__init__
        RepetitionPenaltyLogitsProcessor.__call__ = AdvancedRepetitionPenaltyLogitsProcessor.__call__


        class LuaLogitsProcessor(LogitsProcessor):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                assert scores.ndim == 2
                assert input_ids.ndim == 2
                self.regeneration_required = False
                self.halt = False

                scores_shape = scores.shape
                scores_list = scores.tolist()
                Variables.lua_koboldbridge.logits = Variables.lua_state.table()
                for r, row in enumerate(scores_list):
                    Variables.lua_koboldbridge.logits[r + 1] = Variables.lua_state.table(*row)
                Variables.lua_koboldbridge.vocab_size = scores_shape[-1]

                execute_genmod()

                scores = torch.FloatTensor(
                    tuple(tuple(row.values()) for row in Variables.lua_koboldbridge.logits.values()),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                assert scores.shape == scores_shape

                return scores


        def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
            processors = new_get_logits_processor.old_get_logits_processor(*args, **kwargs)
            processors.insert(0, LuaLogitsProcessor())
            return processors


        new_get_logits_processor.old_get_logits_processor = transformers.generation_utils.GenerationMixin._get_logits_processor
        transformers.generation_utils.GenerationMixin._get_logits_processor = new_get_logits_processor


        def new_get_logits_warper(beams: int = 1, ) -> LogitsProcessorList:
            warper_list = LogitsProcessorList()
            warper_list.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1 + (beams > 1)))
            warper_list.append(TopPLogitsWarper(top_p=0.5, min_tokens_to_keep=1 + (beams > 1)))
            warper_list.append(TailFreeLogitsWarper(tfs=0.5, min_tokens_to_keep=1 + (beams > 1)))
            warper_list.append(TemperatureLogitsWarper(temperature=0.5))
            return warper_list


        def new_sample(self, *args, **kwargs):
            assert kwargs.pop("logits_warper", None) is not None
            kwargs["logits_warper"] = new_get_logits_warper(
                beams=1,
            )
            if Variables.newlinemode == "s":
                kwargs["eos_token_id"] = -1
                kwargs.setdefault("pad_token_id", 2)
            return new_sample.old_sample(self, *args, **kwargs)


        new_sample.old_sample = transformers.generation_utils.GenerationMixin.sample
        transformers.generation_utils.GenerationMixin.sample = new_sample

        # Allow bad words filter to ban <|endoftext|> token
        import transformers.generation_logits_process


        def new_init(self, bad_words_ids: List[List[int]], eos_token_id=-1):
            return new_init.old_init(self, bad_words_ids, -1)


        new_init.old_init = transformers.generation_logits_process.NoBadWordsLogitsProcessor.__init__
        transformers.generation_logits_process.NoBadWordsLogitsProcessor.__init__ = new_init


        # Sets up dynamic world info scanner
        class DynamicWorldInfoScanCriteria(StoppingCriteria):

            def __init__(self, tokenizer, excluded_world_info: List[Set], *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.regeneration_required = False
                self.halt = False
                self.tokenizer = tokenizer
                self.excluded_world_info = excluded_world_info

            def __call__(
                    self,
                    input_ids: torch.LongTensor,
                    scores: torch.FloatTensor,
                    **kwargs,
            ) -> bool:
                Variables.generated_tkns += 1
                if (
                        Variables.lua_koboldbridge.generated_cols and Variables.generated_tkns != Variables.lua_koboldbridge.generated_cols):
                    raise RuntimeError(
                        f"Inconsistency detected between KoboldAI Python and Lua backends ({Variables.generated_tkns} != {Variables.lua_koboldbridge.generated_cols})")
                if Variables.abort or Variables.generated_tkns >= Variables.genamt:
                    self.regeneration_required = False
                    self.halt = False
                    return True

                assert input_ids.ndim == 2
                assert len(self.excluded_world_info) == input_ids.shape[0]
                self.regeneration_required = Variables.lua_koboldbridge.regeneration_required
                self.halt = not Variables.lua_koboldbridge.generating
                Variables.lua_koboldbridge.regeneration_required = False

                for i in range(Variables.numseqs):
                    Variables.lua_koboldbridge.generated[i + 1][Variables.generated_tkns] = int(input_ids[i, -1].item())

                if not Variables.dynamicscan:
                    return self.regeneration_required or self.halt
                tail = input_ids[..., -Variables.generated_tkns:]
                for i, t in enumerate(tail):
                    decoded = utils.decodenewlines(tokenizer.decode(t), Variables.newlinemode)
                    _, found = checkworldinfo(decoded, force_use_txt=True, actions=Variables._actions)
                    found -= self.excluded_world_info[i]
                    if len(found) != 0:
                        self.regeneration_required = True
                        break
                return self.regeneration_required or self.halt


        old_get_stopping_criteria = transformers.generation_utils.GenerationMixin._get_stopping_criteria


        def new_get_stopping_criteria(self, *args, **kwargs):
            stopping_criteria = old_get_stopping_criteria(self, *args, **kwargs)
            global tokenizer
            self.kai_scanner = DynamicWorldInfoScanCriteria(
                tokenizer=tokenizer,
                excluded_world_info=self.kai_scanner_excluded_world_info,
            )
            stopping_criteria.insert(0, self.kai_scanner)
            return stopping_criteria


        transformers.generation_utils.GenerationMixin._get_stopping_criteria = new_get_stopping_criteria


        def get_hidden_size_from_model(model):
            try:
                return int(model.transformer.hidden_size)
            except:
                try:
                    return int(model.transformer.embed_dim)
                except:
                    return int(model.lm_head.in_features)


        def maybe_low_cpu_mem_usage() -> Dict[str, Any]:
            if packaging.__version__.parse(transformers_version) < packaging.__version__.parse("4.11.0"):
                print(
                    f"\nWARNING:  Please upgrade to transformers 4.11.0 for lower RAM usage.  You have transformers {transformers_version}.",
                    file=sys.stderr)
                return {}
            return {"low_cpu_mem_usage": True}


        @contextlib.contextmanager
        def maybe_use_float16(always_use=False):
            if always_use or (Variables.hascuda and args.lowmem and (Variables.usegpu or Variables.breakmodel)):
                original_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float16)
                yield True
                torch.set_default_dtype(original_dtype)
            else:
                yield False


        # If custom GPT2 model was chosen
        if Variables.model == "GPT2Custom":
            Variables.lazy_load = False
            model_config = open(Variables.custmodpth + "/config.json", "r")
            js = json.load(model_config)
            with(maybe_use_float16()):
                model = GPT2LMHeadModel.from_pretrained(Variables.custmodpth, cache_dir="cache/")
            tokenizer = GPT2TokenizerFast.from_pretrained(Variables.custmodpth, cache_dir="cache/")
            Variables.modeldim = get_hidden_size_from_model(model)
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if Variables.hascuda and Variables.usegpu:
                model = model.half().to(Variables.gpu_device)
                generator = model.generate
            else:
                model = model.to('cpu').float()
                generator = model.generate
        # Use the Generic implementation
        else:
            lowmem = maybe_low_cpu_mem_usage()
            # We must disable low_cpu_mem_usage (by setting lowmem to {}) if
            # using a GPT-2 model because GPT-2 is not compatible with this
            # feature yet
            if Variables.model_type == "gpt2":
                lowmem = {}

            # If we're using torch_lazy_loader, we need to get breakmodel config
            # early so that it knows where to load the individual model tensors
            if Variables.lazy_load and Variables.hascuda and Variables.breakmodel:
                device_config(model_config)

            # Download model from Huggingface if it does not exist, otherwise load locally

            # If we specify a model and it's in the root directory, we need to move it to the models directory (legacy folder structure to new)
            if os.path.isdir(Variables.model.replace('/', '_')):
                import shutil

                shutil.move(Variables.model.replace('/', '_'), "models/{}".format(Variables.model.replace('/', '_')))
            with maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(enable=Variables.lazy_load,
                                                                            callback=get_lazy_load_callback(
                                                                                model_config.num_layers if hasattr(
                                                                                    model_config,
                                                                                    "num_layers") else model_config.n_layer),
                                                                            dematerialized_modules=True):
                if Variables.lazy_load:  # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                    lowmem = {}
                if os.path.isdir(Variables.custmodpth):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(Variables.custmodpth, cache_dir="cache")
                    except ValueError as e:
                        tokenizer = GPT2TokenizerFast.from_pretrained(Variables.custmodpth, cache_dir="cache")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(Variables.custmodpth, cache_dir="cache", **lowmem)
                    except ValueError as e:
                        model = GPTNeoForCausalLM.from_pretrained(Variables.custmodpth, cache_dir="cache", **lowmem)
                elif os.path.isdir("models/{}".format(Variables.model.replace('/', '_'))):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained("models/{}".format(Variables.model.replace('/', '_')),
                                                                  cache_dir="cache")
                    except ValueError as e:
                        tokenizer = GPT2TokenizerFast.from_pretrained(
                            "models/{}".format(Variables.model.replace('/', '_')),
                            cache_dir="cache")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            "models/{}".format(Variables.model.replace('/', '_')),
                            cache_dir="cache", **lowmem)
                    except ValueError as e:
                        model = GPTNeoForCausalLM.from_pretrained("models/{}".format(Variables.model.replace('/', '_')),
                                                                  cache_dir="cache", **lowmem)
                else:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(Variables.model, cache_dir="cache")
                    except ValueError as e:
                        tokenizer = GPT2TokenizerFast.from_pretrained(Variables.model, cache_dir="cache")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(Variables.model, cache_dir="cache", **lowmem)
                    except ValueError as e:
                        model = GPTNeoForCausalLM.from_pretrained(Variables.model, cache_dir="cache", **lowmem)

                    if not args.colab:
                        import shutil

                        model = model.half()
                        model.save_pretrained("models/{}".format(Variables.model.replace('/', '_')))
                        tokenizer.save_pretrained("models/{}".format(Variables.model.replace('/', '_')))
                        shutil.rmtree("cache/")

            if Variables.hascuda:
                if Variables.usegpu:
                    Variables.modeldim = get_hidden_size_from_model(model)
                    model = model.half().to(Variables.gpu_device)
                    generator = model.generate
                elif Variables.breakmodel:  # Use both RAM and VRAM (breakmodel)
                    Variables.modeldim = get_hidden_size_from_model(model)
                    if not Variables.lazy_load:
                        device_config(model.config)
                    move_model_to_devices(model)
                else:
                    model = model.to('cpu').float()
                    Variables.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
            else:
                model.to('cpu').float()
                Variables.modeldim = get_hidden_size_from_model(model)
                generator = model.generate

        print("{0}OK! {1} pipeline created!{2}".format(Colors.GREEN, Variables.model, Colors.END))

    else:
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
else:
    def tpumtjgetsofttokens():
        if Variables.sp is None:
            tensor = np.zeros((1, tpu_mtj_backend.params["d_model"]), dtype=np.float32)
            rows = tensor.shape[0]
            padding_amount = tpu_mtj_backend.params["seq"] - (
                    tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
            tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
            tensor = tensor.reshape(
                tpu_mtj_backend.params["cores_per_replica"],
                -1,
                tpu_mtj_backend.params["d_model"],
            )
            Variables.sp = tpu_mtj_backend.shard_xmap(tensor)
        soft_tokens = np.arange(
            tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
            tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + Variables.sp_length,
            dtype=np.uint32
        )
        return soft_tokens


    def tpumtjgenerate_warper_callback(scores) -> "np.array":
        scores_shape = scores.shape
        scores_list = scores.tolist()
        Variables.lua_koboldbridge.logits = Variables.lua_state.table()
        for r, row in enumerate(scores_list):
            Variables.lua_koboldbridge.logits[r + 1] = Variables.lua_state.table(*row)
        Variables.lua_koboldbridge.vocab_size = scores_shape[-1]

        execute_genmod()

        scores = np.array(
            tuple(tuple(row.values()) for row in Variables.lua_koboldbridge.logits.values()),
            dtype=scores.dtype,
        )
        assert scores.shape == scores_shape

        return scores


    def tpumtjgenerate_stopping_callback(generated, n_generated, excluded_world_info) -> Tuple[List[set], bool, bool]:
        Variables.generated_tkns += 1

        assert len(excluded_world_info) == len(generated)
        regeneration_required = Variables.lua_koboldbridge.regeneration_required
        halt = Variables.abort or not Variables.lua_koboldbridge.generating or Variables.generated_tkns >= Variables.genamt
        Variables.lua_koboldbridge.regeneration_required = False

        global past

        for i in range(Variables.numseqs):
            Variables.lua_koboldbridge.generated[i + 1][Variables.generated_tkns] = int(
                generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item())

        if not Variables.dynamicscan or halt:
            return excluded_world_info, regeneration_required, halt

        for i, t in enumerate(generated):
            decoded = utils.decodenewlines(tokenizer.decode(past[i]), Variables.newlinemode) + utils.decodenewlines(
                tokenizer.decode(t[tpu_mtj_backend.params["seq"]: tpu_mtj_backend.params["seq"] + n_generated]),
                Variables.newlinemode)
            _, found = checkworldinfo(decoded, force_use_txt=True, actions=Variables._actions)
            found -= excluded_world_info[i]
            if len(found) != 0:
                regeneration_required = True
                break
        return excluded_world_info, regeneration_required, halt


    def tpumtjgenerate_compiling_callback() -> None:
        print(Colors.GREEN + "TPU backend compilation triggered" + Colors.END)
        Variables.compiling = True


    def tpumtjgenerate_stopped_compiling_callback() -> None:
        Variables.compiling = False


    def tpumtjgenerate_settings_callback() -> dict:
        return {
            "top_p": float(Variables.top_p),
            "temp": float(Variables.temp),
            "top_k": int(Variables.top_k),
            "tfs": float(Variables.tfs),
            "repetition_penalty": float(Variables.rep_pen),
            "rpslope": float(Variables.rep_pen_slope),
            "rprange": int(Variables.rep_pen_range),
        }


    # If we're running Colab or OAI, we still need a tokenizer.
    if Variables.model == "Colab":
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir="cache/")
        loadsettings()
    elif Variables.model == "OAI":
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
        loadsettings()
    # Load the TPU backend if requested
    elif Variables.use_colab_tpu or Variables.model == "TPUMeshTransformerGPTJ":
        print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(Colors.PURPLE, Colors.END))
        if Variables.model == "TPUMeshTransformerGPTJ" and (
                not Variables.custmodpth or not os.path.isdir(Variables.custmodpth)):
            raise FileNotFoundError(
                f"The specified model path {repr(Variables.custmodpth)} is not the path to a valid folder")
        import tpu_mtj_backend

        tpu_mtj_backend.vars = vars
        tpu_mtj_backend.warper_callback = tpumtjgenerate_warper_callback
        tpu_mtj_backend.stopping_callback = tpumtjgenerate_stopping_callback
        tpu_mtj_backend.compiling_callback = tpumtjgenerate_compiling_callback
        tpu_mtj_backend.stopped_compiling_callback = tpumtjgenerate_stopped_compiling_callback
        tpu_mtj_backend.settings_callback = tpumtjgenerate_settings_callback
        Variables.allowsp = True
        loadmodelsettings()
        loadsettings()
        tpu_mtj_backend.load_model(Variables.custmodpth,
                                   hf_checkpoint=Variables.model != "TPUMeshTransformerGPTJ" and Variables.use_colab_tpu,
                                   **Variables.modelconfig)
        Variables.modeldim = int(tpu_mtj_backend.params["d_model"])
        tokenizer = tpu_mtj_backend.tokenizer
    else:
        loadsettings()


# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/download')
def download():
    save_format = request.args.get("format", "json").strip().lower()

    if save_format == "plaintext":
        txt = Variables.prompt + "".join(Variables.actions.values())
        save = Response(txt)
        filename = path.basename(Variables.savedir)
        if filename[-5:] == ".json":
            filename = filename[:-5]
        save.headers.set('Content-Disposition', 'attachment', filename='%s.txt' % filename)
        return save

    # Build json to write
    js = {}
    js["gamestarted"] = Variables.gamestarted
    js["prompt"] = Variables.prompt
    js["memory"] = Variables.memory
    js["authorsnote"] = Variables.authornote
    js["anotetemplate"] = Variables.authornotetemplate
    js["actions"] = tuple(Variables.actions.values())
    js["actions_metadata"] = Variables.actions_metadata
    js["worldinfo"] = []

    # Extract only the important bits of WI
    for wi in Variables.worldinfo:
        if wi["constant"] or wi["key"] != "":
            js["worldinfo"].append({
                "key": wi["key"],
                "keysecondary": wi["keysecondary"],
                "content": wi["content"],
                "comment": wi["comment"],
                "folder": wi["folder"],
                "selective": wi["selective"],
                "constant": wi["constant"]
            })

    save = Response(json.dumps(js, indent=3))
    filename = path.basename(Variables.savedir)
    if filename[-5:] == ".json":
        filename = filename[:-5]
    save.headers.set('Content-Disposition', 'attachment', filename='%s.json' % filename)
    return save


# ============================ LUA API =============================#

if path.exists("settings/" + getmodelname().replace('/', '_') + ".settings"):
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
    js = json.load(file)
    if "userscripts" in js:
        Variables.userscripts = []
        for userscript in js["userscripts"]:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(
                    userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                Variables.userscripts.append(userscript)
    if ("corescript" in js and type(js["corescript"]) is str and all(
            q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
        Variables.corescript = js["corescript"]
    else:
        Variables.corescript = "default.lua"
    file.close()


def lua_log_format_name(name):
    return f"[{name}]" if type(name) is str else "CORE"


_bridged = {}
F = TypeVar("F", bound=Callable)


def bridged_kwarg(name=None):
    def _bridged_kwarg(f: F):
        _bridged[name if name is not None else f.__name__[4:] if f.__name__[:4] == "lua_" else f.__name__] = f
        return f

    return _bridged_kwarg


# ==================================================================#
#  Event triggered when a userscript is loaded
# ==================================================================#
@bridged_kwarg()
def load_callback(filename, modulename):
    print(Colors.GREEN + f"Loading Userscript [{modulename}] <{filename}>" + Colors.END)


# ==================================================================#
#  Load all Lua scripts
# ==================================================================#
def load_lua_scripts():
    print(Colors.GREEN + "Loading Core Script" + Colors.END)

    filenames = []
    modulenames = []
    descriptions = []

    lst = fileops.getusfiles(long_desc=True)
    filenames_dict = {ob["filename"]: i for i, ob in enumerate(lst)}

    for filename in Variables.userscripts:
        if filename in filenames_dict:
            i = filenames_dict[filename]
            filenames.append(filename)
            modulenames.append(lst[i]["modulename"])
            descriptions.append(lst[i]["description"])

    Variables.has_genmod = False

    try:
        Variables.lua_koboldbridge.obliterate_multiverse()
        tpool.execute(Variables.lua_koboldbridge.load_corescript, Variables.corescript)
        Variables.has_genmod = tpool.execute(Variables.lua_koboldbridge.load_userscripts, filenames, modulenames,
                                             descriptions)
        Variables.lua_running = True
    except lupa.LuaError as e:
        try:
            Variables.lua_koboldbridge.obliterate_multiverse()
        except:
            pass
        Variables.lua_running = False
        if Variables.serverstarted:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
            sendusstatitems()
        print("{0}{1}{2}".format(Colors.RED, "***LUA ERROR***: ", Colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(Colors.RED, str(e).replace("\033", ""), Colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(Colors.YELLOW,
                                 "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.",
                                 Colors.END), file=sys.stderr)
        if Variables.serverstarted:
            set_aibusy(0)


# ==================================================================#
#  Print message that originates from the userscript with the given name
# ==================================================================#
@bridged_kwarg()
def lua_print(msg):
    if Variables.lua_logname != Variables.lua_koboldbridge.logging_name:
        Variables.lua_logname = Variables.lua_koboldbridge.logging_name
        print(Colors.BLUE + lua_log_format_name(Variables.lua_logname) + ":" + Colors.END, file=sys.stderr)
    print(Colors.PURPLE + msg.replace("\033", "") + Colors.END)


# ==================================================================#
#  Print warning that originates from the userscript with the given name
# ==================================================================#
@bridged_kwarg()
def lua_warn(msg):
    if Variables.lua_logname != Variables.lua_koboldbridge.logging_name:
        Variables.lua_logname = Variables.lua_koboldbridge.logging_name
        print(Colors.BLUE + lua_log_format_name(Variables.lua_logname) + ":" + Colors.END, file=sys.stderr)
    print(Colors.YELLOW + msg.replace("\033", "") + Colors.END)


# ==================================================================#
#  Decode tokens into a string using current tokenizer
# ==================================================================#
@bridged_kwarg()
def lua_decode(tokens):
    tokens = list(tokens.values())
    assert type(tokens) is list
    if "tokenizer" not in globals():
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
    return utils.decodenewlines(tokenizer.decode(tokens), Variables.newlinemode)


# ==================================================================#
#  Encode string into list of token IDs using current tokenizer
# ==================================================================#
@bridged_kwarg()
def lua_encode(string):
    assert type(string) is str
    if "tokenizer" not in globals():
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
    return tokenizer.encode(utils.encodenewlines(string, Variables.newlinemode), max_length=int(4e9), truncation=True)


# ==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
# ==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if kwargs is None:
        kwargs = Variables.lua_state.table()
    actions = Variables._actions if Variables.lua_koboldbridge.userstate == "genmod" else Variables.actions
    allowed_entries = None
    allowed_folders = None
    if entries is not None:
        allowed_entries = set()
        i = 1
        while entries[i] is not None:
            allowed_entries.add(int(entries[i]))
            i += 1
    if folders is not None:
        allowed_folders = set()
        i = 1
        while folders[i] is not None:
            allowed_folders.add(int(folders[i]))
            i += 1
    winfo, mem, anotetxt, _ = calcsubmitbudgetheader(
        submission,
        allowed_entries=allowed_entries,
        allowed_folders=allowed_folders,
        force_use_txt=True,
        scan_story=kwargs["scan_story"] if kwargs["scan_story"] is not None else True,
    )
    txt, _, _ = calcsubmitbudget(
        len(actions),
        winfo,
        mem,
        anotetxt,
        actions,
    )
    return utils.decodenewlines(tokenizer.decode(txt), Variables.newlinemode)


# ==================================================================#
#  Get property of a world info entry given its UID and property name
# ==================================================================#
@bridged_kwarg()
def lua_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if (uid in Variables.worldinfo_u and k in (
            "key",
            "keysecondary",
            "content",
            "comment",
            "folder",
            "num",
            "selective",
            "constant",
            "uid",
    )):
        return Variables.worldinfo_u[uid][k]


# ==================================================================#
#  Set property of a world info entry given its UID, property name and new value
# ==================================================================#
@bridged_kwarg()
def lua_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in Variables.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "selective",
        "constant",
    )
    if type(Variables.worldinfo_u[uid][k]) is int and type(v) is float:
        v = int(v)
    assert type(Variables.worldinfo_u[uid][k]) is type(v)
    Variables.worldinfo_u[uid][k] = v
    print(
        Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} set {k} of world info entry {uid} to {v}" + Colors.END)


# ==================================================================#
#  Get property of a world info folder given its UID and property name
# ==================================================================#
@bridged_kwarg()
def lua_folder_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if (uid in Variables.wifolders_d and k in (
            "name",
    )):
        return Variables.wifolders_d[uid][k]


# ==================================================================#
#  Set property of a world info folder given its UID, property name and new value
# ==================================================================#
@bridged_kwarg()
def lua_folder_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in Variables.wifolders_d and k in (
        "name",
    )
    if type(Variables.wifolders_d[uid][k]) is int and type(v) is float:
        v = int(v)
    assert type(Variables.wifolders_d[uid][k]) is type(v)
    Variables.wifolders_d[uid][k] = v
    print(
        Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} set {k} of world info folder {uid} to {v}" + Colors.END)


# ==================================================================#
#  Get the "Amount to Generate"
# ==================================================================#
@bridged_kwarg()
def lua_get_genamt():
    return Variables.genamt


# ==================================================================#
#  Set the "Amount to Generate"
# ==================================================================#
@bridged_kwarg()
def lua_set_genamt(genamt):
    assert Variables.lua_koboldbridge.userstate != "genmod" and type(genamt) in (int, float) and genamt >= 0
    print(
        Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} set genamt to {int(genamt)}" + Colors.END)
    Variables.genamt = int(genamt)


# ==================================================================#
#  Get the "Gens Per Action"
# ==================================================================#
@bridged_kwarg()
def lua_get_numseqs():
    return Variables.numseqs


# ==================================================================#
#  Set the "Gens Per Action"
# ==================================================================#
@bridged_kwarg()
def lua_set_numseqs(numseqs):
    assert type(numseqs) in (int, float) and numseqs >= 1
    print(
        Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} set numseqs to {int(numseqs)}" + Colors.END)
    Variables.numseqs = int(numseqs)


# ==================================================================#
#  Check if a setting exists with the given name
# ==================================================================#
@bridged_kwarg()
def lua_has_setting(setting):
    return setting in (
        "anotedepth",
        "settemp",
        "settopp",
        "settopk",
        "settfs",
        "setreppen",
        "setreppenslope",
        "setreppenrange",
        "settknmax",
        "setwidepth",
        "setuseprompt",
        "setadventure",
        "setchatmode",
        "setdynamicscan",
        "setnopromptgen",
        "setrngpersist",
        "temp",
        "topp",
        "top_p",
        "topk",
        "top_k",
        "tfs",
        "reppen",
        "reppenslope",
        "reppenrange",
        "tknmax",
        "widepth",
        "useprompt",
        "chatmode",
        "chatname",
        "adventure",
        "dynamicscan",
        "nopromptgen",
        "rngpersist",
        "frmttriminc",
        "frmtrmblln",
        "frmtrmspch",
        "frmtadsnsp",
        "frmtsingleline",
        "triminc",
        "rmblln",
        "rmspch",
        "adsnsp",
        "singleline",
    )


# ==================================================================#
#  Return the setting with the given name if it exists
# ==================================================================#
@bridged_kwarg()
def lua_get_setting(setting):
    if setting in ("settemp", "temp"):
        return Variables.temp
    if setting in ("settopp", "topp", "top_p"):
        return Variables.top_p
    if setting in ("settopk", "topk", "top_k"):
        return Variables.top_k
    if setting in ("settfs", "tfs"):
        return Variables.tfs
    if setting in ("setreppen", "reppen"):
        return Variables.rep_pen
    if setting in ("setreppenslope", "reppenslope"):
        return Variables.rep_pen_slope
    if setting in ("setreppenrange", "reppenrange"):
        return Variables.rep_pen_range
    if setting in ("settknmax", "tknmax"):
        return Variables.max_length
    if setting == "anotedepth":
        return Variables.andepth
    if setting in ("setwidepth", "widepth"):
        return Variables.widepth
    if setting in ("setuseprompt", "useprompt"):
        return Variables.useprompt
    if setting in ("setadventure", "adventure"):
        return Variables.adventure
    if setting in ("setchatmode", "chatmode"):
        return Variables.chatmode
    if setting in ("setdynamicscan", "dynamicscan"):
        return Variables.dynamicscan
    if setting in ("setnopromptgen", "nopromptgen"):
        return Variables.nopromptgen
    if setting in ("setrngpersist", "rngpersist"):
        return Variables.rngpersist
    if setting in ("frmttriminc", "triminc"):
        return Variables.formatoptns["frmttriminc"]
    if setting in ("frmtrmblln", "rmblln"):
        return Variables.formatoptns["frmttrmblln"]
    if setting in ("frmtrmspch", "rmspch"):
        return Variables.formatoptns["frmttrmspch"]
    if setting in ("frmtadsnsp", "adsnsp"):
        return Variables.formatoptns["frmtadsnsp"]
    if setting in ("frmtsingleline", "singleline"):
        return Variables.formatoptns["singleline"]


# ==================================================================#
#  Set the setting with the given name if it exists
# ==================================================================#
@bridged_kwarg()
def lua_set_setting(setting, v):
    actual_type = type(lua_get_setting(setting))
    assert v is not None and (actual_type is type(v) or (actual_type is int and type(v) is float))
    v = actual_type(v)
    print(
        Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} set {setting} to {v}" + Colors.END)
    if setting in ("setadventure", "adventure") and v:
        Variables.actionmode = 1
    if setting in ("settemp", "temp"):
        Variables.temp = v
    if setting in ("settopp", "topp"):
        Variables.top_p = v
    if setting in ("settopk", "topk"):
        Variables.top_k = v
    if setting in ("settfs", "tfs"):
        Variables.tfs = v
    if setting in ("setreppen", "reppen"):
        Variables.rep_pen = v
    if setting in ("setreppenslope", "reppenslope"):
        Variables.rep_pen_slope = v
    if setting in ("setreppenrange", "reppenrange"):
        Variables.rep_pen_range = v
    if setting in ("settknmax", "tknmax"):
        Variables.max_length = v
        return True
    if setting == "anotedepth":
        Variables.andepth = v
        return True
    if setting in ("setwidepth", "widepth"):
        Variables.widepth = v
        return True
    if setting in ("setuseprompt", "useprompt"):
        Variables.useprompt = v
        return True
    if setting in ("setadventure", "adventure"):
        Variables.adventure = v
    if setting in ("setdynamicscan", "dynamicscan"):
        Variables.dynamicscan = v
    if setting in ("setnopromptgen", "nopromptgen"):
        Variables.nopromptgen = v
    if setting in ("setrngpersist", "rngpersist"):
        Variables.rngpersist = v
    if setting in ("setchatmode", "chatmode"):
        Variables.chatmode = v
    if setting in ("frmttriminc", "triminc"):
        Variables.formatoptns["frmttriminc"] = v
    if setting in ("frmtrmblln", "rmblln"):
        Variables.formatoptns["frmttrmblln"] = v
    if setting in ("frmtrmspch", "rmspch"):
        Variables.formatoptns["frmttrmspch"] = v
    if setting in ("frmtadsnsp", "adsnsp"):
        Variables.formatoptns["frmtadsnsp"] = v
    if setting in ("frmtsingleline", "singleline"):
        Variables.formatoptns["singleline"] = v


# ==================================================================#
#  Get contents of memory
# ==================================================================#
@bridged_kwarg()
def lua_get_memory():
    return Variables.memory


# ==================================================================#
#  Set contents of memory
# ==================================================================#
@bridged_kwarg()
def lua_set_memory(m):
    assert type(m) is str
    Variables.memory = m


# ==================================================================#
#  Get contents of author's note
# ==================================================================#
@bridged_kwarg()
def lua_get_authorsnote():
    return Variables.authornote


# ==================================================================#
#  Set contents of author's note
# ==================================================================#
@bridged_kwarg()
def lua_set_authorsnote(m):
    assert type(m) is str
    Variables.authornote = m


# ==================================================================#
#  Get contents of author's note template
# ==================================================================#
@bridged_kwarg()
def lua_get_authorsnotetemplate():
    return Variables.authornotetemplate


# ==================================================================#
#  Set contents of author's note template
# ==================================================================#
@bridged_kwarg()
def lua_set_authorsnotetemplate(m):
    assert type(m) is str
    Variables.authornotetemplate = m


# ==================================================================#
#  Save settings and send them to client
# ==================================================================#
@bridged_kwarg()
def lua_resend_settings():
    settingschanged()
    refresh_settings()


# ==================================================================#
#  Set story chunk text and delete the chunk if the new chunk is empty
# ==================================================================#
@bridged_kwarg()
def lua_set_chunk(k, v):
    assert type(k) in (int, None) and type(v) is str
    assert k >= 0
    assert k != 0 or len(v) != 0
    if len(v) == 0:
        print(
            Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} deleted story chunk {k}" + Colors.END)
        chunk = int(k)
        if Variables.lua_koboldbridge.userstate == "genmod":
            del Variables._actions[chunk - 1]
        Variables.lua_deleted.add(chunk)
        if not hasattr(vars, "_actions") or Variables._actions is not Variables.actions:
            # Instead of deleting we'll blank out the text. This way our actions and actions_metadata stay in sync and we can restore the chunk on an undo
            Variables.actions[chunk - 1] = ""
            Variables.actions_metadata[chunk - 1]['Alternative Text'] = [{"Text": Variables.actions_metadata[chunk - 1][
                'Selected Text'], "Pinned": False, "Editted": True}] + Variables.actions_metadata[chunk - 1][
                                                                            'Alternative Text']
            Variables.actions_metadata[chunk - 1]['Selected Text'] = ''
            send_debug()
    else:
        if k == 0:
            print(
                Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} edited prompt chunk" + Colors.END)
        else:
            print(
                Colors.GREEN + f"{lua_log_format_name(Variables.lua_koboldbridge.logging_name)} edited story chunk {k}" + Colors.END)
        chunk = int(k)
        if chunk == 0:
            if Variables.lua_koboldbridge.userstate == "genmod":
                Variables._prompt = v
            Variables.lua_edited.add(chunk)
            Variables.prompt = v
        else:
            if Variables.lua_koboldbridge.userstate == "genmod":
                Variables._actions[chunk - 1] = v
            Variables.lua_edited.add(chunk)
            Variables.actions[chunk - 1] = v
            Variables.actions_metadata[chunk - 1]['Alternative Text'] = [{"Text": Variables.actions_metadata[chunk - 1][
                'Selected Text'], "Pinned": False, "Editted": True}] + Variables.actions_metadata[chunk - 1][
                                                                            'Alternative Text']
            Variables.actions_metadata[chunk - 1]['Selected Text'] = v
            send_debug()


# ==================================================================#
#  Get model type as "gpt-2-xl", "gpt-neo-2.7B", etc.
# ==================================================================#
@bridged_kwarg()
def lua_get_modeltype():
    global hidden_size
    if Variables.noai:
        return "readonly"
    if Variables.model in ("Colab", "OAI", "InferKit"):
        return "api"
    if (not Variables.use_colab_tpu and Variables.model not in ("TPUMeshTransformerGPTJ",) and (
            Variables.model in ("GPT2Custom", "NeoCustom") or Variables.model_type in ("gpt2", "gpt_neo", "gptj"))):
        hidden_size = get_hidden_size_from_model(model)
    if Variables.model in ("gpt2",) or (Variables.model_type == "gpt2" and hidden_size == 768):
        return "gpt2"
    if Variables.model in ("gpt2-medium",) or (Variables.model_type == "gpt2" and hidden_size == 1024):
        return "gpt2-medium"
    if Variables.model in ("gpt2-large",) or (Variables.model_type == "gpt2" and hidden_size == 1280):
        return "gpt2-large"
    if Variables.model in ("gpt2-xl",) or (Variables.model_type == "gpt2" and hidden_size == 1600):
        return "gpt2-xl"
    if Variables.model_type == "gpt_neo" and hidden_size == 768:
        return "gpt-neo-125M"
    if Variables.model in ("EleutherAI/gpt-neo-1.3B",) or (Variables.model_type == "gpt_neo" and hidden_size == 2048):
        return "gpt-neo-1.3B"
    if Variables.model in ("EleutherAI/gpt-neo-2.7B",) or (Variables.model_type == "gpt_neo" and hidden_size == 2560):
        return "gpt-neo-2.7B"
    if (Variables.model in ("EleutherAI/gpt-j-6B",) or (
            (Variables.use_colab_tpu or Variables.model == "TPUMeshTransformerGPTJ") and
              tpu_mtj_backend.params["d_model"] == 4096) or
              (Variables.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)):
        return "gpt-j-6B"
    return "unknown"


# ==================================================================#
#  Get model backend as "transformers" or "mtj"
# ==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if Variables.noai:
        return "readonly"
    if Variables.model in ("Colab", "OAI", "InferKit"):
        return "api"
    if Variables.use_colab_tpu or Variables.model in ("TPUMeshTransformerGPTJ",):
        return "mtj"
    return "transformers"


# ==================================================================#
#  Check whether model is loaded from a custom path
# ==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return Variables.model in ("GPT2Custom", "NeoCustom", "TPUMeshTransformerGPTJ")


# ==================================================================#
#  
# ==================================================================#
def execute_inmod():
    setgamesaved(False)
    Variables.lua_logname = ...
    Variables.lua_edited = set()
    Variables.lua_deleted = set()
    try:
        tpool.execute(Variables.lua_koboldbridge.execute_inmod)
    except lupa.LuaError as e:
        Variables.lua_koboldbridge.obliterate_multiverse()
        Variables.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
        sendusstatitems()
        print("{0}{1}{2}".format(Colors.RED, "***LUA ERROR***: ", Colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(Colors.RED, str(e).replace("\033", ""), Colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(Colors.YELLOW,
                                 "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.",
                                 Colors.END), file=sys.stderr)
        set_aibusy(0)


def execute_genmod():
    Variables.lua_koboldbridge.execute_genmod()


def execute_outmod():
    setgamesaved(False)
    emit('from_server', {'cmd': 'hidemsg', 'data': ''}, broadcast=True)
    try:
        tpool.execute(Variables.lua_koboldbridge.execute_outmod)
    except lupa.LuaError as e:
        Variables.lua_koboldbridge.obliterate_multiverse()
        Variables.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
        sendusstatitems()
        print("{0}{1}{2}".format(Colors.RED, "***LUA ERROR***: ", Colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(Colors.RED, str(e).replace("\033", ""), Colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(Colors.YELLOW,
                                 "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.",
                                 Colors.END), file=sys.stderr)
        set_aibusy(0)
    if Variables.lua_koboldbridge.resend_settings_required:
        Variables.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()
    for k in Variables.lua_edited:
        inlineedit(k, Variables.actions[k])
    for k in Variables.lua_deleted:
        inlinedelete(k)


# ==================================================================#
#  Lua runtime startup
# ==================================================================#

print("", end="", flush=True)
print(Colors.PURPLE + "Initializing Lua Bridge... " + Colors.END, end="", flush=True)

# Set up Lua state
Variables.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

# Load bridge.lua
bridged = {
    "corescript_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "cores"),
    "userscript_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "userscripts"),
    "config_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "userscripts"),
    "lib_paths": Variables.lua_state.table(os.path.join(os.path.dirname(os.path.realpath(__file__)), "lualibs"),
                                           os.path.join(os.path.dirname(os.path.realpath(__file__)), "extern",
                                                        "lualibs")),
    "vars": vars,
}
for kwarg in _bridged:
    bridged[kwarg] = _bridged[kwarg]
try:
    Variables.lua_kobold, Variables.lua_koboldcore, Variables.lua_koboldbridge = Variables.lua_state.globals().dofile(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "bridge.lua"))(
        Variables.lua_state.globals().python,
        bridged,
    )
except lupa.LuaError as e:
    print(Colors.RED + "ERROR!" + Colors.END)
    Variables.lua_koboldbridge.obliterate_multiverse()
    print("{0}{1}{2}".format(Colors.RED, "***LUA ERROR***: ", Colors.END), end="", file=sys.stderr)
    print("{0}{1}{2}".format(Colors.RED, str(e).replace("\033", ""), Colors.END), file=sys.stderr)
    exit(1)
print(Colors.GREEN + "OK!" + Colors.END)

# Load scripts
load_lua_scripts()


# ============================ METHODS =============================#

# ==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
# ==================================================================#
@socketio.on('connect')
def do_connect():
    print("{0}Client connected!{1}".format(Colors.GREEN, Colors.END))
    emit('from_server', {'cmd': 'setchatname', 'data': Variables.chatname})
    emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate})
    emit('from_server', {'cmd': 'connected', 'smandelete': Variables.smandelete, 'smanrename': Variables.smanrename,
                         'modelname': getmodelname()})
    if Variables.host:
        emit('from_server', {'cmd': 'runs_remotely'})
    if Variables.allowsp:
        emit('from_server', {'cmd': 'allowsp', 'data': Variables.allowsp})

    sendusstatitems()
    emit('from_server', {'cmd': 'spstatitems',
                         'data': {Variables.spfilename: Variables.spmeta} if Variables.allowsp and len(
                             Variables.spfilename) else {}},
         broadcast=True)

    if not Variables.gamestarted:
        setstartstate()
        sendsettings()
        refresh_settings()
        Variables.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory})
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory})
        emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote})
        Variables.mode = "play"
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        sendsettings()
        refresh_settings()
        emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory})
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory})
        emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote})
        if Variables.mode == "play":
            if not Variables.aibusy:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'})
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'})
        elif Variables.mode == "edit":
            emit('from_server', {'cmd': 'editmode', 'data': 'true'})
        elif Variables.mode == "memory":
            emit('from_server', {'cmd': 'memmode', 'data': 'true'})
        elif Variables.mode == "wi":
            emit('from_server', {'cmd': 'wimode', 'data': 'true'})

    emit('from_server', {'cmd': 'gamesaved', 'data': Variables.gamesaved}, broadcast=True)


# ==================================================================#
# Event triggered when browser SocketIO sends data to the server
# ==================================================================#
@socketio.on('message')
def get_message(msg):
    if not Variables.quiet:
        print("{0}Data received:{1}{2}".format(Colors.GREEN, msg, Colors.END))
    # Submit action
    if msg['cmd'] == 'submit':
        if Variables.mode == "play":
            if Variables.aibusy:
                if msg.get('allowabort', False):
                    Variables.abort = True
                return
            Variables.abort = False
            Variables.lua_koboldbridge.feedback = None
            if Variables.chatmode:
                if type(msg['chatname']) is not str:
                    raise ValueError("Chatname must be a string")
                Variables.chatname = msg['chatname']
                settingschanged()
                emit('from_server', {'cmd': 'setchatname', 'data': Variables.chatname})
            Variables.recentrng = Variables.recentrngm = None
            actionsubmit(msg['data'], actionmode=msg['actionmode'])
        elif Variables.mode == "edit":
            editsubmit(msg['data'])
        elif Variables.mode == "memory":
            memsubmit(msg['data'])
    # Retry Action
    elif msg['cmd'] == 'retry':
        if Variables.aibusy:
            if msg.get('allowabort', False):
                Variables.abort = True
            return
        Variables.abort = False
        if Variables.chatmode:
            if type(msg['chatname']) is not str:
                raise ValueError("Chatname must be a string")
            Variables.chatname = msg['chatname']
            settingschanged()
            emit('from_server', {'cmd': 'setchatname', 'data': Variables.chatname})
        actionretry()
    # Back/Undo Action
    elif msg['cmd'] == 'back':
        actionback()
    # Forward/Redo Action
    elif msg['cmd'] == 'redo':
        actionredo()
    # EditMode Action (old)
    elif msg['cmd'] == 'edit':
        if Variables.mode == "play":
            Variables.mode = "edit"
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, broadcast=True)
        elif Variables.mode == "edit":
            Variables.mode = "play"
            emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    # EditLine Action (old)
    elif msg['cmd'] == 'editline':
        editrequest(int(msg['data']))
    # Inline edit
    elif msg['cmd'] == 'inlineedit':
        inlineedit(msg['chunk'], msg['data'])
    elif msg['cmd'] == 'inlinedelete':
        inlinedelete(msg['data'])
    # DeleteLine Action (old)
    elif msg['cmd'] == 'delete':
        deleterequest()
    elif msg['cmd'] == 'memory':
        togglememorymode()
    elif not Variables.host and msg['cmd'] == 'savetofile':
        savetofile()
    elif not Variables.host and msg['cmd'] == 'loadfromfile':
        loadfromfile()
    elif msg['cmd'] == 'loadfromstring':
        loadrequest(json.loads(msg['data']), filename=msg['filename'])
    elif not Variables.host and msg['cmd'] == 'import':
        importrequest()
    elif msg['cmd'] == 'newgame':
        newgamerequest()
    elif msg['cmd'] == 'rndgame':
        randomgamerequest(msg['data'], memory=msg['memory'])
    elif msg['cmd'] == 'settemp':
        Variables.temp = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltemp', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'settopp':
        Variables.top_p = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopp', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'settopk':
        Variables.top_k = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopk', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'settfs':
        Variables.tfs = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltfs', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setreppen':
        Variables.rep_pen = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppen', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setreppenslope':
        Variables.rep_pen_slope = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenslope', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setreppenrange':
        Variables.rep_pen_range = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenrange', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setoutput':
        Variables.genamt = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeloutput', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'settknmax':
        Variables.max_length = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltknmax', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setikgen':
        Variables.ikgen = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelikgen', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    # Author's Note field update
    elif msg['cmd'] == 'anote':
        anotesubmit(msg['data'], template=msg['template'])
    # Author's Note depth update
    elif msg['cmd'] == 'anotedepth':
        Variables.andepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelanotedepth', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    # Format - Trim incomplete sentences
    elif msg['cmd'] == 'frmttriminc':
        if 'frmttriminc' in Variables.formatoptns:
            Variables.formatoptns["frmttriminc"] = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'frmtrmblln':
        if 'frmtrmblln' in Variables.formatoptns:
            Variables.formatoptns["frmtrmblln"] = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'frmtrmspch':
        if 'frmtrmspch' in Variables.formatoptns:
            Variables.formatoptns["frmtrmspch"] = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'frmtadsnsp':
        if 'frmtadsnsp' in Variables.formatoptns:
            Variables.formatoptns["frmtadsnsp"] = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'singleline':
        if 'singleline' in Variables.formatoptns:
            Variables.formatoptns["singleline"] = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'importselect':
        Variables.importnum = int(msg["data"].replace("import", ""))
    elif msg['cmd'] == 'importcancel':
        emit('from_server', {'cmd': 'popupshow', 'data': False})
        Variables.importjs = {}
    elif msg['cmd'] == 'importaccept':
        emit('from_server', {'cmd': 'popupshow', 'data': False})
        importgame()
    elif msg['cmd'] == 'wi':
        togglewimode()
    elif msg['cmd'] == 'wiinit':
        if int(msg['data']) < len(Variables.worldinfo):
            setgamesaved(False)
            Variables.worldinfo[msg['data']]["init"] = True
            addwiitem(folder_uid=msg['folder'])
    elif msg['cmd'] == 'wifolderinit':
        addwifolder()
    elif msg['cmd'] == 'wimoveitem':
        movewiitem(msg['destination'], msg['data'])
    elif msg['cmd'] == 'wimovefolder':
        movewifolder(msg['destination'], msg['data'])
    elif msg['cmd'] == 'widelete':
        deletewi(msg['data'])
    elif msg['cmd'] == 'wifolderdelete':
        deletewifolder(msg['data'])
    elif msg['cmd'] == 'wiexpand':
        assert 0 <= int(msg['data']) < len(Variables.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpand', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wiexpandfolder':
        assert 0 <= int(msg['data']) < len(Variables.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpandfolder', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wifoldercollapsecontent':
        setgamesaved(False)
        Variables.wifolders_d[msg['data']]['collapsed'] = True
        emit('from_server', {'cmd': 'wifoldercollapsecontent', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wifolderexpandcontent':
        setgamesaved(False)
        Variables.wifolders_d[msg['data']]['collapsed'] = False
        emit('from_server', {'cmd': 'wifolderexpandcontent', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wiupdate':
        setgamesaved(False)
        num = int(msg['num'])
        fields = ("key", "keysecondary", "content", "comment")
        for field in fields:
            if field in msg['data'] and type(msg['data'][field]) is str:
                Variables.worldinfo[num][field] = msg['data'][field]
        emit('from_server',
             {'cmd': 'wiupdate', 'num': msg['num'],
              'data': {field: Variables.worldinfo[num][field] for field in fields}},
             broadcast=True)
    elif msg['cmd'] == 'wifolderupdate':
        setgamesaved(False)
        uid = int(msg['uid'])
        fields = ("name", "collapsed")
        for field in fields:
            if field in msg['data'] and type(msg['data'][field]) is (str if field != "collapsed" else bool):
                Variables.wifolders_d[uid][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wifolderupdate', 'uid': msg['uid'],
                             'data': {field: Variables.wifolders_d[uid][field] for field in fields}}, broadcast=True)
    elif msg['cmd'] == 'wiselon':
        setgamesaved(False)
        Variables.worldinfo[msg['data']]["selective"] = True
        emit('from_server', {'cmd': 'wiselon', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wiseloff':
        setgamesaved(False)
        Variables.worldinfo[msg['data']]["selective"] = False
        emit('from_server', {'cmd': 'wiseloff', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wiconstanton':
        setgamesaved(False)
        Variables.worldinfo[msg['data']]["constant"] = True
        emit('from_server', {'cmd': 'wiconstanton', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'wiconstantoff':
        setgamesaved(False)
        Variables.worldinfo[msg['data']]["constant"] = False
        emit('from_server', {'cmd': 'wiconstantoff', 'data': msg['data']}, broadcast=True)
    elif msg['cmd'] == 'sendwilist':
        commitwi(msg['data'])
    elif msg['cmd'] == 'aidgimport':
        importaidgrequest(msg['data'])
    elif msg['cmd'] == 'saveasrequest':
        saveas(msg['data'])
    elif msg['cmd'] == 'saverequest':
        save()
    elif msg['cmd'] == 'loadlistrequest':
        getloadlist()
    elif msg['cmd'] == 'splistrequest':
        getsplist()
    elif msg['cmd'] == 'uslistrequest':
        unloaded, loaded = getuslist()
        emit('from_server', {'cmd': 'buildus', 'data': {"unloaded": unloaded, "loaded": loaded}})
    elif msg['cmd'] == 'usloaded':
        Variables.userscripts = []
        for userscript in msg['data']:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(
                    userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                Variables.userscripts.append(userscript)
        settingschanged()
    elif msg['cmd'] == 'usload':
        load_lua_scripts()
        _, _ = getuslist()
        sendusstatitems()
    elif msg['cmd'] == 'loadselect':
        Variables.loadselect = msg["data"]
    elif msg['cmd'] == 'spselect':
        Variables.spselect = msg["data"]
    elif msg['cmd'] == 'loadrequest':
        loadrequest(fileops.storypath(Variables.loadselect))
    elif msg['cmd'] == 'sprequest':
        sprequest(Variables.spselect)
        emit('from_server', {'cmd': 'spstatitems',
                             'data': {Variables.spfilename: Variables.spmeta} if Variables.allowsp and len(
                                 Variables.spfilename) else {}},
             broadcast=True)
    elif msg['cmd'] == 'deletestory':
        deletesave(msg['data'])
    elif msg['cmd'] == 'renamestory':
        renamesave(msg['data'], msg['newname'])
    elif msg['cmd'] == 'clearoverwrite':
        Variables.svowname = ""
        Variables.saveow = False
    elif msg['cmd'] == 'seqsel':
        selectsequence(msg['data'])
    elif msg['cmd'] == 'seqpin':
        pinsequence(msg['data'])
    elif msg['cmd'] == 'setnumseq':
        Variables.numseqs = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelnumseq', 'data': msg['data']})
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setwidepth':
        Variables.widepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelwidepth', 'data': msg['data']})
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setuseprompt':
        Variables.useprompt = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setadventure':
        Variables.adventure = msg['data']
        Variables.chatmode = False
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'autosave':
        Variables.autosave = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setchatmode':
        Variables.chatmode = msg['data']
        Variables.adventure = False
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setdynamicscan':
        Variables.dynamicscan = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setnopromptgen':
        Variables.nopromptgen = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setrngpersist':
        Variables.rngpersist = msg['data']
        settingschanged()
        refresh_settings()
    elif msg['cmd'] == 'setnogenmod':
        Variables.nogenmod = msg['data']
        settingschanged()
        refresh_settings()
    elif not Variables.host and msg['cmd'] == 'importwi':
        wiimportrequest()
    elif msg['cmd'] == 'debug':
        Variables.debug = msg['data']
        emit('from_server', {'cmd': 'set_debug', 'data': msg['data']}, broadcast=True)
        if Variables.debug:
            send_debug()


# ==================================================================#
#  Send userscripts list to client
# ==================================================================#
def sendusstatitems():
    _, loaded = getuslist()
    loaded = loaded if Variables.lua_running else []
    last_userscripts = [e["filename"] for e in loaded]
    emit('from_server', {'cmd': 'usstatitems', 'data': loaded, 'flash': last_userscripts != Variables.last_userscripts},
         broadcast=True)
    Variables.last_userscripts = last_userscripts


# ==================================================================#
#  KoboldAI Markup Formatting (Mixture of Markdown and sanitized html)
# ==================================================================#
def kml(txt):
    txt = txt.replace('>', '&gt;')
    txt = bleach.clean(markdown.markdown(txt),
                       tags=['p', 'em', 'strong', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'b', 'i', 'a',
                             'span', 'button'], styles=['color', 'font-weight'],
                       attributes=['id', 'class', 'style', 'href'])
    return txt


# ==================================================================#
#  Send start message and tell Javascript to set UI state
# ==================================================================#
def setstartstate():
    if Variables.welcome:
        txt = kml(Variables.welcome) + "<br/>"
    else:
        txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">" + getmodelname() + "</span>.<br/>"
    if not Variables.noai and not Variables.welcome:
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    if Variables.noai:
        txt = txt + "Please load or import a story to read. There is no AI in this mode."
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': Variables.gamestarted, 'data': txt}, broadcast=True)
    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'}, broadcast=True)


# ==================================================================#
#  Transmit applicable settings to SocketIO to build UI sliders/toggles
# ==================================================================#
def sendsettings():
    # Send settings for selected AI type
    if Variables.model != "InferKit":
        for setting in gensettings.gensettingstf:
            emit('from_server', {'cmd': 'addsetting', 'data': setting})
    else:
        for setting in gensettings.gensettingsik:
            emit('from_server', {'cmd': 'addsetting', 'data': setting})

    # Send formatting options
    for frm in gensettings.formatcontrols:
        emit('from_server', {'cmd': 'addformat', 'data': frm})
        # Add format key to vars if it wasn't loaded with client.settings
        if not frm["id"] in Variables.formatoptns:
            Variables.formatoptns[frm["id"]] = False


# ==================================================================#
#  Set value of gamesaved
# ==================================================================#
def setgamesaved(gamesaved):
    assert type(gamesaved) is bool
    if gamesaved != Variables.gamesaved:
        emit('from_server', {'cmd': 'gamesaved', 'data': gamesaved}, broadcast=True)
    Variables.gamesaved = gamesaved


# ==================================================================#
#  Take input text from SocketIO and decide what to do with it
# ==================================================================#

def check_for_backend_compilation():
    if Variables.checking:
        return
    Variables.checking = True
    for _ in range(31):
        time.sleep(0.06276680299820175)
        if Variables.compiling:
            emit('from_server',
                 {'cmd': 'warnmsg', 'data': 'Compiling TPU backend&mdash;this usually takes 1&ndash;2 minutes...'},
                 broadcast=True)
            break
    Variables.checking = False


def actionsubmit(data, actionmode=0, force_submit=False, force_prompt_gen=False, disable_recentrng=False):
    # Ignore new submissions if the AI is currently busy
    if Variables.aibusy:
        return

    while True:
        set_aibusy(1)

        if disable_recentrng:
            Variables.recentrng = Variables.recentrngm = None

        Variables.recentback = False
        Variables.recentedit = False
        Variables.actionmode = actionmode

        # "Action" mode
        if actionmode == 1:
            data = data.strip().lstrip('>')
            data = re.sub(r'\n+', ' ', data)
            if len(data):
                data = f"\n\n> {data}\n"

        # "Chat" mode
        if Variables.chatmode and Variables.gamestarted:
            data = re.sub(r'\n+', ' ', data)
            if len(data):
                data = f"\n{Variables.chatname}: {data}\n"

        # If we're not continuing, store a copy of the raw input
        if data != "":
            Variables.lastact = data

        if not Variables.gamestarted:
            Variables.submission = data
            execute_inmod()
            data = Variables.submission
            if not force_submit and len(data.strip()) == 0:
                assert False
            # Start the game
            Variables.gamestarted = True
            if not Variables.noai and Variables.lua_koboldbridge.generating and (
                    not Variables.nopromptgen or force_prompt_gen):
                # Save this first action as the prompt
                Variables.prompt = data
                # Clear the startup text from game screen
                emit('from_server',
                     {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'},
                     broadcast=True)
                calcsubmit(data)  # Run the first action through the generator
                if not Variables.abort and Variables.lua_koboldbridge.restart_sequence is not None and len(
                        Variables.genseqs) == 0:
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break
            else:
                # Save this first action as the prompt
                Variables.prompt = data if len(data) > 0 else '"'
                for i in range(Variables.numseqs):
                    Variables.lua_koboldbridge.outputs[i + 1] = ""
                execute_outmod()
                Variables.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(Variables.numseqs):
                    genout.append({"generated_text": Variables.lua_koboldbridge.outputs[i + 1]})
                    assert type(genout[-1]["generated_text"]) is str
                if len(genout) == 1:
                    genresult(genout[0]["generated_text"], flash=False)
                    refresh_story()
                    if len(Variables.actions) > 0:
                        emit('from_server', {'cmd': 'texteffect', 'data': Variables.actions.get_last_key() + 1},
                             broadcast=True)
                    if not Variables.abort and Variables.lua_koboldbridge.restart_sequence is not None:
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if (
                            not Variables.abort and Variables.lua_koboldbridge.restart_sequence is not None and Variables.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[Variables.lua_koboldbridge.restart_sequence - 1]["generated_text"],
                                  flash=False)
                        refresh_story()
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                    refresh_story()
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break
        else:
            # Apply input formatting & scripts before sending to tokenizer
            if Variables.actionmode == 0:
                data = applyinputformatting(data)
            Variables.submission = data
            execute_inmod()
            data = Variables.submission
            # Dont append submission if it's a blank/continue action
            if data != "":
                # Store the result in the Action log
                if len(Variables.prompt.strip()) == 0:
                    Variables.prompt = data
                else:
                    Variables.actions.append(data)
                    # we now need to update the actions_metadata
                    # we'll have two conditions. 
                    # 1. This is totally new (user entered) 
                    if Variables.actions.get_last_key() not in Variables.actions_metadata:
                        Variables.actions_metadata[Variables.actions.get_last_key()] = {"Selected Text": data,
                                                                                        "Alternative Text": []}
                    else:
                        # 2. We've selected a chunk of text that is was presented previously
                        try:
                            alternatives = [item['Text'] for item in
                                            Variables.actions_metadata[len(Variables.actions) - 1]["Alternative Text"]]
                        except:
                            print(len(Variables.actions))
                            print(Variables.actions_metadata)
                            raise
                        if data in alternatives:
                            alternatives = [item for item in
                                            Variables.actions_metadata[Variables.actions.get_last_key()][
                                                "Alternative Text"] if
                                            item['Text'] != data]
                            Variables.actions_metadata[Variables.actions.get_last_key()][
                                "Alternative Text"] = alternatives
                        Variables.actions_metadata[Variables.actions.get_last_key()]["Selected Text"] = data
                update_story_chunk('last')
                send_debug()

            if not Variables.noai and Variables.lua_koboldbridge.generating:
                # Off to the tokenizer!
                calcsubmit(data)
                if not Variables.abort and Variables.lua_koboldbridge.restart_sequence is not None and len(
                        Variables.genseqs) == 0:
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break
            else:
                for i in range(Variables.numseqs):
                    Variables.lua_koboldbridge.outputs[i + 1] = ""
                execute_outmod()
                Variables.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(Variables.numseqs):
                    genout.append({"generated_text": Variables.lua_koboldbridge.outputs[i + 1]})
                    assert type(genout[-1]["generated_text"]) is str
                if len(genout) == 1:
                    genresult(genout[0]["generated_text"])
                    if not Variables.abort and Variables.lua_koboldbridge.restart_sequence is not None:
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if (
                            not Variables.abort and Variables.lua_koboldbridge.restart_sequence is not None and Variables.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[Variables.lua_koboldbridge.restart_sequence - 1]["generated_text"])
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break


# ==================================================================#
#  
# ==================================================================#
def actionretry():
    if Variables.noai:
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."})
        return
    if Variables.aibusy:
        return
    if Variables.recentrng is not None:
        randomgamerequest(Variables.recentrng, memory=Variables.recentrngm)
        return
    # Remove last action if possible and resubmit
    if Variables.gamestarted if Variables.useprompt else len(Variables.actions) > 0:
        if (not Variables.recentback and len(Variables.actions) != 0 and len(
                Variables.genseqs) == 0):  # Don't pop if we're in the "Select sequence to keep" menu or if there are no non-prompt actions
            # We are going to move the selected text to alternative text in the actions_metadata variable so we can redo this action
            Variables.actions_metadata[Variables.actions.get_last_key()]['Alternative Text'] = \
                [{'Text': Variables.actions_metadata[Variables.actions.get_last_key()]['Selected Text'],
                  'Pinned': False, "Previous Selection": True, "Edited": False}] + \
                    Variables.actions_metadata[Variables.actions.get_last_key()]['Alternative Text']
            Variables.actions_metadata[Variables.actions.get_last_key()]['Selected Text'] = ""

            last_key = Variables.actions.get_last_key()
            Variables.actions.pop()
            remove_story_chunk(last_key + 1)
        Variables.recentback = False
        Variables.recentedit = False
        Variables.lua_koboldbridge.feedback = None
        actionsubmit("", actionmode=Variables.actionmode, force_submit=True)
        send_debug()
    elif not Variables.useprompt:
        emit('from_server', {'cmd': 'errmsg', 'data': "Please enable \"Always Add Prompt\" to retry with your prompt."})


# ==================================================================#
#  
# ==================================================================#
def actionback():
    if Variables.aibusy:
        return
    # Remove last index of actions and refresh game screen
    if len(Variables.genseqs) == 0 and len(Variables.actions) > 0:
        # We are going to move the selected text to alternative text in the actions_metadata variable so we can redo this action
        last_action =  Variables.actions_metadata[Variables.actions.get_last_key()]['Alternative Text']
        Variables.actions_metadata[Variables.actions.get_last_key()]['Alternative Text'] = [{'Text':
          Variables.actions_metadata[Variables.actions.get_last_key()]['Selected Text'],'Pinned': False,
          "Previous Selection": True,"Edited": False}] + last_action
        Variables.actions_metadata[Variables.actions.get_last_key()]['Selected Text'] = ""

        last_key = Variables.actions.get_last_key()
        Variables.actions.pop()
        Variables.recentback = True
        remove_story_chunk(last_key + 1)
        # for the redo to not get out of whack, need to reset the max # in the actions sequence
        Variables.actions.set_next_id(last_key)
    elif len(Variables.genseqs) == 0:
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."})
    else:
        Variables.genseqs = []
    send_debug()


def actionredo():
    # First we need to find the next valid key
    # We might have deleted text so we don't want to show a redo for that blank chunk

    restore_id = Variables.actions.get_last_key() + 1
    if restore_id in Variables.actions_metadata:
        ok_to_use = False
        while not ok_to_use:
            for item in Variables.actions_metadata[restore_id]['Alternative Text']:
                if item['Previous Selection'] and item['Text'] != "":
                    ok_to_use = True
            if not ok_to_use:
                restore_id += 1
                if restore_id not in Variables.actions_metadata:
                    return
            else:
                Variables.actions.set_next_id(restore_id)

    if restore_id in Variables.actions_metadata:
        genout = [{"generated_text": item['Text']} for item in
                  Variables.actions_metadata[restore_id]['Alternative Text'] if
                  (item["Previous Selection"] is True)]
        if len(genout) > 0:
            genout = genout + [{"generated_text": item['Text']} for item in
                               Variables.actions_metadata[restore_id]['Alternative Text'] if
                               (item["Pinned"] is True) and (item["Previous Selection"] is False)]
            if len(genout) == 1:
                Variables.actions_metadata[restore_id]['Alternative Text'] = [item for item in
                                                                              Variables.actions_metadata[restore_id][
                                                                                  'Alternative Text'] if
                                                                              (item["Previous Selection"] is not True)]
                genresult(genout[0]['generated_text'], flash=True, ignore_formatting=True)
            else:
                # Store sequences in memory until selection is made
                Variables.genseqs = genout

                # Send sequences to UI for selection
                genout = [[item['Text'], "redo"] for item in Variables.actions_metadata[restore_id]['Alternative Text']
                          if
                          (item["Previous Selection"] is True)]

                emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True)
    else:
        emit('from_server', {'cmd': 'popuperror', 'data': "There's nothing to undo"}, broadcast=True)
    send_debug()


# ==================================================================#
#  
# ==================================================================#
def calcsubmitbudgetheader(txt, **kwargs):
    # Scan for WorldInfo matches
    winfo, found_entries = checkworldinfo(txt, **kwargs)

    # Add a newline to the end of memory
    if Variables.memory != "" and Variables.memory[-1] != "\n":
        mem = Variables.memory + "\n"
    else:
        mem = Variables.memory

    # Build Author's Note if set
    if Variables.authornote != "":
        anotetxt = ("\n" + Variables.authornotetemplate + "\n").replace("<|>", Variables.authornote)
    else:
        anotetxt = ""

    return winfo, mem, anotetxt, found_entries


def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions, submission=None, budget_deduction=0):
    forceanote = False  # In case we don't have enough actions to hit A.N. depth
    anoteadded = False  # In case our budget runs out before we hit A.N. depth
    anotetkns = []  # Placeholder for Author's Note tokens
    lnanote = 0  # Placeholder for Author's Note length

    lnsp = Variables.sp_length

    if "tokenizer" not in globals():
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")

    # Calculate token budget
    prompttkns = tokenizer.encode(
        utils.encodenewlines(Variables.comregex_ai.sub('', Variables.prompt), Variables.newlinemode),
        max_length=int(2e9),
        truncation=True)
    lnprompt = len(prompttkns)

    memtokens = tokenizer.encode(utils.encodenewlines(mem, Variables.newlinemode), max_length=int(2e9), truncation=True)
    lnmem = len(memtokens)
    if lnmem > Variables.max_length - lnsp - Variables.genamt - budget_deduction:
        raise OverflowError(
            "The memory in your story is too long. Please either write a shorter memory text or increase the Max "
            "Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    witokens = tokenizer.encode(utils.encodenewlines(winfo, Variables.newlinemode), max_length=int(2e9),
                                truncation=True)
    lnwi = len(witokens)
    if lnmem + lnwi > Variables.max_length - lnsp - Variables.genamt - budget_deduction:
        raise OverflowError(
            "The current active world info keys take up too many tokens. Please either write shorter world info, "
            "decrease World Info Depth or increase the Max Tokens setting. If you are using a soft prompt, "
            "additionally consider using a smaller soft prompt.")

    if anotetxt != "":
        anotetkns = tokenizer.encode(utils.encodenewlines(anotetxt, Variables.newlinemode), max_length=int(2e9),
                                     truncation=True)
        lnanote = len(anotetkns)
        if lnmem + lnwi + lnanote > Variables.max_length - lnsp - Variables.genamt - budget_deduction:
            raise OverflowError(
                "The author's note in your story is too long. Please either write a shorter author's note or increase "
                "the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft "
                "prompt.")

    if Variables.useprompt:
        budget = Variables.max_length - lnsp - lnprompt - lnmem - lnanote - lnwi - Variables.genamt - budget_deduction
    else:
        budget = Variables.max_length - lnsp - lnmem - lnanote - lnwi - Variables.genamt - budget_deduction

    lnsubmission = len(
        tokenizer.encode(utils.encodenewlines(Variables.comregex_ai.sub('', submission), Variables.newlinemode),
                         max_length=int(2e9),
                         truncation=True)) if submission is not None else 0
    maybe_lnprompt = lnprompt if Variables.useprompt and actionlen > 0 else 0

    if (
            lnmem + lnwi + lnanote + maybe_lnprompt + lnsubmission > Variables.max_length - lnsp -
            Variables.genamt - budget_deduction):
        raise OverflowError(
            "Your submission is too long. Please either write a shorter submission or increase the Max Tokens "
            "setting. If you are using a soft prompt, additionally consider using a smaller soft prompt. If you are "
            "using the Always Add Prompt setting, turning it off may help.")

    assert budget >= 0

    if actionlen == 0:
        # First/Prompt action
        tokens = memtokens + witokens + anotetkns + prompttkns
        assert len(tokens) <= Variables.max_length - lnsp - Variables.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln + 1, ln + Variables.genamt
    else:
        tokens = []

        # Check if we have the action depth to hit our A.N. depth
        if anotetxt != "" and actionlen < Variables.andepth:
            forceanote = True

        # Get most recent action tokens up to our budget
        n = 0
        for key in reversed(actions):
            chunk = Variables.comregex_ai.sub('', actions[key])

            assert budget >= 0
            if budget <= 0:
                break
            acttkns = tokenizer.encode(utils.encodenewlines(chunk, Variables.newlinemode), max_length=int(2e9),
                                       truncation=True)
            tknlen = len(acttkns)
            if tknlen < budget:
                tokens = acttkns + tokens
                budget -= tknlen
            else:
                count = budget * -1
                tokens = acttkns[count:] + tokens
                budget = 0
                break

            # Inject Author's Note if we've reached the desired depth
            if n == Variables.andepth - 1:
                if anotetxt != "":
                    tokens = anotetkns + tokens  # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1

        # If we're not using the prompt every time and there's still budget left,
        # add some prompt.
        if not Variables.useprompt:
            if budget > 0:
                prompttkns = prompttkns[-budget:]
            else:
                prompttkns = []

        # Did we get to add the A.N.? If not, do it here
        if anotetxt != "":
            if (not anoteadded) or forceanote:
                tokens = memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = memtokens + witokens + prompttkns + tokens

        # Send completed bundle to generator
        assert len(tokens) <= Variables.max_length - lnsp - Variables.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln + 1, ln + Variables.genamt


# ==================================================================#
# Take submitted text and build the text to be given to generator
# ==================================================================#
def calcsubmit(txt):
    forceanote = False  # In case we don't have enough actions to hit A.N. depth
    anoteadded = False  # In case our budget runs out before we hit A.N. depth
    actionlen = len(Variables.actions)

    winfo, mem, anotetxt, found_entries = calcsubmitbudgetheader(txt)

    # For all transformers models
    if Variables.model != "InferKit":
        subtxt, mini, maxi = calcsubmitbudget(actionlen, winfo, mem, anotetxt, Variables.actions, submission=txt)
        if actionlen == 0:
            if not Variables.use_colab_tpu and Variables.model not in ["Colab", "OAI", "TPUMeshTransformerGPTJ"]:
                generate(subtxt, mini, maxi, found_entries=found_entries)
            elif Variables.model == "Colab":
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt), Variables.newlinemode), mini, maxi)
            elif Variables.model == "OAI":
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt), Variables.newlinemode), maxi)
            elif Variables.use_colab_tpu or Variables.model == "TPUMeshTransformerGPTJ":
                tpumtjgenerate(subtxt, mini, maxi, found_entries=found_entries)
        else:
            if not Variables.use_colab_tpu and Variables.model not in ["Colab", "OAI", "TPUMeshTransformerGPTJ"]:
                generate(subtxt, mini, maxi, found_entries=found_entries)
            elif Variables.model == "Colab":
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt), Variables.newlinemode), mini, maxi)
            elif Variables.model == "OAI":
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt), Variables.newlinemode), maxi)
            elif Variables.use_colab_tpu or Variables.model == "TPUMeshTransformerGPTJ":
                tpumtjgenerate(subtxt, mini, maxi, found_entries=found_entries)

    # For InferKit web API
    else:
        # Check if we have the action depth to hit our A.N. depth
        if anotetxt != "" and actionlen < Variables.andepth:
            forceanote = True

        if Variables.useprompt:
            budget = Variables.ikmax - len(Variables.comregex_ai.sub('', Variables.prompt)) - len(anotetxt) - len(
                mem) - len(winfo) - 1
        else:
            budget = Variables.ikmax - len(anotetxt) - len(mem) - len(winfo) - 1

        subtxt = ""
        prompt = Variables.comregex_ai.sub('', Variables.prompt)
        n = 0
        for key in reversed(Variables.actions):
            chunk = Variables.actions[key]

            if budget <= 0:
                break
            actlen = len(chunk)
            if actlen < budget:
                subtxt = chunk + subtxt
                budget -= actlen
            else:
                count = budget * -1
                subtxt = chunk[count:] + subtxt
                break

            # If we're not using the prompt every time and there's still budget left,
            # add some prompt.
            if not Variables.useprompt:
                if budget > 0:
                    prompt = Variables.comregex_ai.sub('', Variables.prompt)[-budget:]
                else:
                    prompt = ""

            # Inject Author's Note if we've reached the desired depth
            if n == Variables.andepth - 1:
                if anotetxt != "":
                    subtxt = anotetxt + subtxt  # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1

        # Did we get to add the A.N.? If not, do it here
        if anotetxt != "":
            if (not anoteadded) or forceanote:
                subtxt = mem + winfo + anotetxt + prompt + subtxt
            else:
                subtxt = mem + winfo + prompt + subtxt
        else:
            subtxt = mem + winfo + prompt + subtxt

        # Send it!
        ikrequest(subtxt)


# ==================================================================#
# Send text to generator and deal with output
# ==================================================================#

def _generate(txt, minimum, maximum, found_entries):
    gen_in = torch.tensor(txt, dtype=torch.long)[None]
    if Variables.sp is not None:
        soft_tokens = torch.arange(
            model.config.vocab_size,
            model.config.vocab_size + Variables.sp.shape[0],
        )
        gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)
    assert gen_in.shape[-1] + Variables.genamt <= Variables.max_length

    if Variables.hascuda and Variables.usegpu:
        gen_in = gen_in.to(Variables.gpu_device)
    elif Variables.hascuda and Variables.breakmodel:
        gen_in = gen_in.to(breakmodel.primary_device)
    else:
        gen_in = gen_in.to('cpu')

    model.kai_scanner_excluded_world_info = found_entries

    Variables._actions = Variables.actions
    Variables._prompt = Variables.prompt
    if Variables.dynamicscan:
        Variables._actions = Variables._actions.copy()

    with torch.no_grad():
        already_generated = 0
        numseqs = Variables.numseqs
        while True:
            genout = generator(
                gen_in,
                do_sample=True,
                max_length=int(2e9),
                repetition_penalty=1.1,
                bad_words_ids=Variables.badwordsids,
                use_cache=True,
                num_return_sequences=numseqs
            )
            already_generated += len(genout[0]) - len(gen_in[0])
            assert already_generated <= Variables.genamt
            if model.kai_scanner.halt or not model.kai_scanner.regeneration_required:
                break
            assert genout.ndim >= 2
            assert genout.shape[0] == Variables.numseqs
            if Variables.lua_koboldbridge.generated_cols and Variables.generated_tkns != Variables.lua_koboldbridge.generated_cols:
                raise RuntimeError("Inconsistency detected between KoboldAI Python and Lua backends")
            if already_generated != Variables.generated_tkns:
                raise RuntimeError("WI scanning error")
            for r in range(Variables.numseqs):
                for c in range(already_generated):
                    assert Variables.lua_koboldbridge.generated[r + 1][c + 1] is not None
                    genout[r][genout.shape[-1] - already_generated + c] = Variables.lua_koboldbridge.generated[r + 1][
                        c + 1]
            encoded = []
            for i in range(Variables.numseqs):
                txt = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]), Variables.newlinemode)
                winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True,
                                                                              actions=Variables._actions)
                found_entries[i].update(_found_entries)
                txt, _, _ = calcsubmitbudget(len(Variables._actions), winfo, mem, anotetxt, Variables._actions,
                                             submission=txt)
                encoded.append(torch.tensor(txt, dtype=torch.long, device=genout.device))
            max_length = len(max(encoded, key=len))
            encoded = torch.stack(tuple(torch.nn.functional.pad(e, (max_length - len(e), 0),
                                                                value=model.config.pad_token_id or model.config.eos_token_id)
                                        for e in encoded))
            genout = torch.cat(
                (
                    encoded,
                    genout[..., -already_generated:],
                ),
                dim=-1
            )
            if Variables.sp is not None:
                soft_tokens = torch.arange(
                    model.config.vocab_size,
                    model.config.vocab_size + Variables.sp.shape[0],
                    device=genout.device,
                )
                genout = torch.cat((soft_tokens.tile(Variables.numseqs, 1), genout), dim=-1)
            assert genout.shape[-1] + Variables.genamt - already_generated <= Variables.max_length
            diff = genout.shape[-1] - gen_in.shape[-1]
            minimum += diff
            maximum += diff
            gen_in = genout
            numseqs = 1

    return genout, already_generated


def generate(txt, minimum, maximum, found_entries=None):
    Variables.generated_tkns = 0

    if found_entries is None:
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(Variables.numseqs))

    if not Variables.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(Colors.YELLOW, minimum, maximum,
                                                       utils.decodenewlines(tokenizer.decode(txt),
                                                                            Variables.newlinemode), Colors.END))

    # Store context in memory to use it for comparison with generated content
    Variables.lastctx = utils.decodenewlines(tokenizer.decode(txt), Variables.newlinemode)

    # Clear CUDA cache if using GPU
    if Variables.hascuda and (Variables.usegpu or Variables.breakmodel):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    try:
        genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, found_entries)
    except Exception as e:
        if issubclass(type(e), lupa.LuaError):
            Variables.lua_koboldbridge.obliterate_multiverse()
            Variables.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
            sendusstatitems()
            print("{0}{1}{2}".format(Colors.RED, "***LUA ERROR***: ", Colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(Colors.RED, str(e).replace("\033", ""), Colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(Colors.YELLOW,
                                     "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.",
                                     Colors.END), file=sys.stderr)
        else:
            emit('from_server',
                 {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'},
                 broadcast=True)
            print("{0}{1}{2}".format(Colors.RED, traceback.format_exc().replace("\033", ""), Colors.END),
                  file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(Variables.numseqs):
        Variables.lua_koboldbridge.generated[i + 1][Variables.generated_tkns] = int(genout[i, -1].item())
        Variables.lua_koboldbridge.outputs[i + 1] = utils.decodenewlines(
            tokenizer.decode(genout[i, -already_generated:]), Variables.newlinemode)

    execute_outmod()
    if Variables.lua_koboldbridge.regeneration_required:
        Variables.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(Variables.numseqs):
            genout.append({"generated_text": Variables.lua_koboldbridge.outputs[i + 1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]),
                                                          Variables.newlinemode)} for tokens in
                  genout]

    if len(genout) == 1:
        genresult(genout[0]["generated_text"])
    else:
        if Variables.lua_koboldbridge.restart_sequence is not None and Variables.lua_koboldbridge.restart_sequence > 0:
            genresult(genout[Variables.lua_koboldbridge.restart_sequence - 1]["generated_text"])
        else:
            genselect(genout)

    # Clear CUDA cache again if using GPU
    if Variables.hascuda and (Variables.usegpu or Variables.breakmodel):
        del genout
        gc.collect()
        torch.cuda.empty_cache()

    set_aibusy(0)


# ==================================================================#
#  Deal with a single return sequence from generate()
# ==================================================================#
def genresult(genout, flash=True, ignore_formatting=False):
    if not Variables.quiet:
        print("{0}{1}{2}".format(Colors.CYAN, genout, Colors.END))

    # Format output before continuing
    if not ignore_formatting:
        genout = applyoutputformatting(genout)

    Variables.lua_koboldbridge.feedback = genout

    if len(genout) == 0:
        return

    # Add formatted text to Actions array and refresh the game screen
    if len(Variables.prompt.strip()) == 0:
        Variables.prompt = genout
    else:
        Variables.actions.append(genout)
        if Variables.actions.get_last_key() not in Variables.actions_metadata:
            Variables.actions_metadata[Variables.actions.get_last_key()] = {'Selected Text': genout,
                                                                            'Alternative Text': []}
        else:
            Variables.actions_metadata[Variables.actions.get_last_key()]['Selected Text'] = genout
    update_story_chunk('last')
    if flash:
        emit('from_server',
             {'cmd': 'texteffect', 'data': Variables.actions.get_last_key() + 1 if len(Variables.actions) else 0},
             broadcast=True)
    send_debug()


# ==================================================================#
#  Send generator sequences to the UI for selection
# ==================================================================#
def genselect(genout):
    i = 0
    for result in genout:
        # Apply output formatting rules to sequences
        result["generated_text"] = applyoutputformatting(result["generated_text"])
        if not Variables.quiet:
            print("{0}[Result {1}]\n{2}{3}".format(Colors.CYAN, i, result["generated_text"], Colors.END))
        i += 1

    # Add the options to the actions metadata
    # If we've already generated text for this action but haven't selected one we'll want to kill all non-pinned, non-previous selection, and non-edited options then add the new ones
    if Variables.actions.get_next_id() in Variables.actions_metadata:
        if Variables.actions_metadata[Variables.actions.get_next_id()]['Selected Text'] == "":
            Variables.actions_metadata[Variables.actions.get_next_id()]['Alternative Text'] = [{"Text": item['Text'],
                                                                                                "Pinned": item[
                                                                                                    'Pinned'],
                                                                                                "Previous Selection":
                                                                                                    item[
                                                                                                        "Previous Selection"],
                                                                                                "Edited": item[
                                                                                                    "Edited"]} for item
                                                                                               in
                                                                                               Variables.actions_metadata[
                                                                                                   Variables.actions.get_next_id()][
                                                                                                   'Alternative Text']
                                                                                               if
                                                                                               item['Pinned'] or item[
                                                                                                   "Previous Selection"] or
                                                                                               item[
                                                                                                   "Edited"]] + [
                                                                                                  {"Text": text[
                                                                                                      "generated_text"],
                                                                                                   "Pinned": False,
                                                                                                   "Previous Selection": False,
                                                                                                   "Edited": False} for
                                                                                                  text in
                                                                                                  genout]
        else:
            Variables.actions_metadata[Variables.actions.get_next_id()] = {'Selected Text': '', 'Alternative Text': [
                {"Text": text["generated_text"], "Pinned": False, "Previous Selection": False, "Edited": False} for text
                in genout]}
    else:
        Variables.actions_metadata[Variables.actions.get_next_id()] = {'Selected Text': '', 'Alternative Text': [
            {"Text": text["generated_text"], "Pinned": False, "Previous Selection": False, "Edited": False} for text in
            genout]}

    genout = [{"generated_text": item['Text']} for item in
              Variables.actions_metadata[Variables.actions.get_next_id()]['Alternative Text'] if
              (item["Previous Selection"] is False) and (item["Edited"] is False)]

    # Store sequences in memory until selection is made
    Variables.genseqs = genout

    genout = [[item['Text'], "pinned" if item['Pinned'] else "normal"] for item in
              Variables.actions_metadata[Variables.actions.get_next_id()]['Alternative Text'] if
              (item["Previous Selection"] is False) and (item["Edited"] is False)]

    # Send sequences to UI for selection
    emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True)
    send_debug()


# ==================================================================#
#  Send selected sequence to action log and refresh UI
# ==================================================================#
def selectsequence(n):
    if len(Variables.genseqs) == 0:
        return
    Variables.lua_koboldbridge.feedback = Variables.genseqs[int(n)]["generated_text"]
    if len(Variables.lua_koboldbridge.feedback) != 0:
        Variables.actions.append(Variables.lua_koboldbridge.feedback)
        # We'll want to remove the option from the alternative text and put it in selected text
        Variables.actions_metadata[Variables.actions.get_last_key()]['Alternative Text'] = [item for item in
                                                                                            Variables.actions_metadata[
                                                                                                Variables.actions.get_last_key()][
                                                                                                'Alternative Text'] if
                                                                                            item[
                                                                                                'Text'] != Variables.lua_koboldbridge.feedback]
        Variables.actions_metadata[Variables.actions.get_last_key()][
            'Selected Text'] = Variables.lua_koboldbridge.feedback
        update_story_chunk('last')
        emit('from_server',
             {'cmd': 'texteffect', 'data': Variables.actions.get_last_key() + 1 if len(Variables.actions) else 0},
             broadcast=True)
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
    Variables.genseqs = []

    if Variables.lua_koboldbridge.restart_sequence is not None:
        actionsubmit("", actionmode=Variables.actionmode, force_submit=True, disable_recentrng=True)
    send_debug()


# ==================================================================#
#  Pin/Unpin the selected sequence
# ==================================================================#
def pinsequence(n):
    if n.isnumeric():
        text = Variables.genseqs[int(n)]['generated_text']
        if text in [item['Text'] for item in
                    Variables.actions_metadata[Variables.actions.get_next_id()]['Alternative Text']]:
            alternatives = Variables.actions_metadata[Variables.actions.get_next_id()]['Alternative Text']
            for i in range(len(alternatives)):
                if alternatives[i]['Text'] == text:
                    alternatives[i]['Pinned'] = not alternatives[i]['Pinned']
                    break
            Variables.actions_metadata[Variables.actions.get_next_id()]['Alternative Text'] = alternatives
    send_debug()


# ==================================================================#
#  Send transformers-style request to ngrok/colab host
# ==================================================================#
def sendtocolab(txt, mini, maxi):
    # Log request to console
    if not Variables.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(Colors.YELLOW, mini - 1, txt, Colors.END))

    # Store context in memory to use it for comparison with generated content
    Variables.lastctx = txt

    # Build request JSON data
    reqdata = {
        'text': txt,
        'min': mini,
        'max': maxi,
        'rep_pen': Variables.rep_pen,
        'rep_pen_slope': Variables.rep_pen_slope,
        'rep_pen_range': Variables.rep_pen_range,
        'temperature': Variables.temp,
        'top_p': Variables.top_p,
        'top_k': Variables.top_k,
        'tfs': Variables.tfs,
        'numseqs': Variables.numseqs,
        'retfultxt': False
    }

    # Create request
    req = requests.post(
        Variables.colaburl,
        json=reqdata
    )

    # Deal with the response
    if req.status_code == 200:
        js = req.json()["data"]

        # Try to be backwards compatible with outdated colab
        if "text" in js:
            genout = [getnewcontent(js["text"])]
        else:
            genout = js["seqs"]

        for i in range(Variables.numseqs):
            Variables.lua_koboldbridge.outputs[i + 1] = genout[i]

        execute_outmod()
        if Variables.lua_koboldbridge.regeneration_required:
            Variables.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(Variables.numseqs):
                genout.append(Variables.lua_koboldbridge.outputs[i + 1])
                assert type(genout[-1]) is str

        if len(genout) == 1:
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            if Variables.lua_koboldbridge.restart_sequence is not None and Variables.lua_koboldbridge.restart_sequence > 0:
                genresult(genout[Variables.lua_koboldbridge.restart_sequence - 1]["generated_text"])
            else:
                genselect(genout)

        # Format output before continuing
        # genout = applyoutputformatting(getnewcontent(genout))

        # Add formatted text to Actions array and refresh the game screen
        # Variables.actions.append(genout)
        # refresh_story()
        # emit('from_server', {'cmd': 'texteffect', 'data': Variables.actions.get_last_key() + 1 if len(Variables.actions) else 0})

        set_aibusy(0)
    else:
        errmsg = "Colab API Error: Failed to get a reply from the server. Please check the colab console."
        print("{0}{1}{2}".format(Colors.RED, errmsg, Colors.END))
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)


# ==================================================================#
#  Send text to TPU mesh transformer backend
# ==================================================================#
def tpumtjgenerate(txt, minimum, maximum, found_entries=None):
    Variables.generated_tkns = 0

    if found_entries is None:
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(Variables.numseqs))

    if not Variables.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(Colors.YELLOW, minimum, maximum,
                                                       utils.decodenewlines(tokenizer.decode(txt),
                                                                            Variables.newlinemode), Colors.END))

    Variables._actions = Variables.actions
    Variables._prompt = Variables.prompt
    if Variables.dynamicscan:
        Variables._actions = Variables._actions.copy()

    # Submit input text to generator
    try:
        soft_tokens = tpumtjgetsofttokens()

        global past

        socketio.start_background_task(copy_current_request_context(check_for_backend_compilation))

        if Variables.dynamicscan or (not Variables.nogenmod and Variables.has_genmod):

            context = np.tile(np.uint32(txt), (Variables.numseqs, 1))
            past = np.empty((Variables.numseqs, 0), dtype=np.uint32)

            while True:
                genout, n_generated, regeneration_required, halt = tpool.execute(
                    tpu_mtj_backend.infer_dynamic,
                    context,
                    gen_len=maximum - minimum + 1,
                    numseqs=Variables.numseqs,
                    soft_embeddings=Variables.sp,
                    soft_tokens=soft_tokens,
                    excluded_world_info=found_entries,
                )

                past = np.pad(past, ((0, 0), (0, n_generated)))
                for r in range(Variables.numseqs):
                    for c in range(Variables.lua_koboldbridge.generated_cols):
                        assert Variables.lua_koboldbridge.generated[r + 1][c + 1] is not None
                        past[r, c] = Variables.lua_koboldbridge.generated[r + 1][c + 1]

                if Variables.abort or halt or not regeneration_required:
                    break
                print("(regeneration triggered)")

                encoded = []
                for i in range(Variables.numseqs):
                    txt = utils.decodenewlines(tokenizer.decode(past[i]), Variables.newlinemode)
                    winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True,
                                                                                  actions=Variables._actions)
                    found_entries[i].update(_found_entries)
                    txt, _, _ = calcsubmitbudget(len(Variables._actions), winfo, mem, anotetxt, Variables._actions,
                                                 submission=txt)
                    encoded.append(np.array(txt, dtype=np.uint32))
                max_length = len(max(encoded, key=len))
                encoded = np.stack(tuple(
                    np.pad(e, (max_length - len(e), 0), constant_values=tpu_mtj_backend.pad_token_id) for e in encoded))
                context = np.concatenate(
                    (
                        encoded,
                        past,
                    ),
                    axis=-1,
                )

        else:
            genout = tpool.execute(
                tpu_mtj_backend.infer_static,
                np.uint32(txt),
                gen_len=maximum - minimum + 1,
                temp=Variables.temp,
                top_p=Variables.top_p,
                top_k=Variables.top_k,
                tfs=Variables.tfs,
                numseqs=Variables.numseqs,
                repetition_penalty=Variables.rep_pen,
                rpslope=Variables.rep_pen_slope,
                rprange=Variables.rep_pen_range,
                soft_embeddings=Variables.sp,
                soft_tokens=soft_tokens,
            )
            past = genout
            for i in range(Variables.numseqs):
                Variables.lua_koboldbridge.generated[i + 1] = Variables.lua_state.table(*genout[i].tolist())
            Variables.lua_koboldbridge.generated_cols = Variables.generated_tkns = genout[0].shape[-1]

    except Exception as e:
        if issubclass(type(e), lupa.LuaError):
            Variables.lua_koboldbridge.obliterate_multiverse()
            Variables.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
            sendusstatitems()
            print("{0}{1}{2}".format(Colors.RED, "***LUA ERROR***: ", Colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(Colors.RED, str(e).replace("\033", ""), Colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(Colors.YELLOW,
                                     "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.",
                                     Colors.END), file=sys.stderr)
        else:
            emit('from_server',
                 {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'},
                 broadcast=True)
            print("{0}{1}{2}".format(Colors.RED, traceback.format_exc().replace("\033", ""), Colors.END),
                  file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(Variables.numseqs):
        Variables.lua_koboldbridge.outputs[i + 1] = utils.decodenewlines(tokenizer.decode(past[i]),
                                                                         Variables.newlinemode)
    genout = past

    execute_outmod()
    if Variables.lua_koboldbridge.regeneration_required:
        Variables.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(Variables.numseqs):
            genout.append({"generated_text": Variables.lua_koboldbridge.outputs[i + 1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(txt), Variables.newlinemode)} for txt in
                  genout]

    if len(genout) == 1:
        genresult(genout[0]["generated_text"])
    else:
        if Variables.lua_koboldbridge.restart_sequence is not None and Variables.lua_koboldbridge.restart_sequence > 0:
            genresult(genout[Variables.lua_koboldbridge.restart_sequence - 1]["generated_text"])
        else:
            genselect(genout)

    set_aibusy(0)


# ==================================================================#
# Replaces returns and newlines with HTML breaks
# ==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r\\n", "<br/>").replace("\\r", "<br/>").replace("\\n", "<br/>").replace("\r\n",
                                                                                                  "<br/>").replace('\n',
                                                                                                                   '<br/>').replace(
        '\r', '<br/>').replace('&lt;/s&gt;', '<br/>')


# ==================================================================#
# Strips submitted text from the text returned by the AI
# ==================================================================#
def getnewcontent(txt):
    # If the submitted context was blank, then everything is new
    if Variables.lastctx == "":
        return txt

    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(utils.encodenewlines(Variables.lastctx, Variables.newlinemode), max_length=int(2e9),
                                 truncation=True)
    txttokens = tokenizer.encode(utils.encodenewlines(txt, Variables.newlinemode), max_length=int(2e9), truncation=True)
    dif = (len(txttokens) - len(ctxtokens)) * -1

    # Remove the context from the returned text
    newtokens = txttokens[dif:]

    return utils.decodenewlines(tokenizer.decode(newtokens), Variables.newlinemode)


# ==================================================================#
# Applies chosen formatting options to text submitted to AI
# ==================================================================#
def applyinputformatting(txt):
    # Add sentence spacing
    if Variables.formatoptns["frmtadsnsp"]:
        txt = utils.addsentencespacing(txt, vars)

    return txt


# ==================================================================#
# Applies chosen formatting options to text returned from AI
# ==================================================================#
def applyoutputformatting(txt):
    # Use standard quotes and apostrophes
    txt = utils.fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if Variables.adventure:
        txt = Variables.acregex_ai.sub('', txt)

    # Trim incomplete sentences
    if Variables.formatoptns["frmttriminc"] and not Variables.chatmode:
        txt = utils.trimincompletesentence(txt)
    # Replace blank lines
    if Variables.formatoptns["frmtrmblln"] or Variables.chatmode:
        txt = utils.replaceblanklines(txt)
    # Remove special characters
    if Variables.formatoptns["frmtrmspch"]:
        txt = utils.removespecialchars(txt, vars)
    # Single Line Mode
    if Variables.formatoptns["singleline"] or Variables.chatmode:
        txt = utils.singlelineprocessing(txt, vars)

    return txt


# ==================================================================#
# Sends the current story content to the Game Screen
# ==================================================================#
def refresh_story():
    text_parts = ['<chunk n="0" id="n0" tabindex="-1">', Variables.comregex_ui.sub(
        lambda m: '\n'.join('<comment>' + comment + '</comment>' for comment in m.group().split('\n')),
        html.escape(Variables.prompt)),
                  '</chunk>']
    for idx in Variables.actions:
        item = Variables.actions[idx]
        idx += 1
        item = html.escape(item)
        item = Variables.comregex_ui.sub(
            lambda m: '\n'.join('<comment>' + comment + '</comment>' for comment in m.group().split('\n')),
            item)  # Add special formatting to comments
        item = Variables.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions
        text_parts.extend(('<chunk n="', str(idx), '" id="n', str(idx), '" tabindex="-1">', item, '</chunk>'))
    emit('from_server',
         {'cmd': 'updatescreen', 'gamestarted': Variables.gamestarted, 'data': formatforhtml(''.join(text_parts))},
         broadcast=True)


# ==================================================================#
# Signals the Game Screen to update one of the chunks
# ==================================================================#
def update_story_chunk(idx: Union[int, str]):
    if idx == 'last':
        if len(Variables.actions) <= 1:
            # In this case, we are better off just refreshing the whole thing as the
            # prompt might not have been shown yet (with a "Generating story..."
            # message instead).
            refresh_story()
            setgamesaved(False)
            return

        idx = (Variables.actions.get_last_key() if len(Variables.actions) else 0) + 1

    if idx == 0:
        text = Variables.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        if idx - 1 not in Variables.actions:
            return
        text = Variables.actions[idx - 1]

    item = html.escape(text)
    item = Variables.comregex_ui.sub(
        lambda m: '\n'.join('<comment>' + comment + '</comment>' for comment in m.group().split('\n')),
        item)  # Add special formatting to comments
    item = Variables.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text}}, broadcast=True)

    setgamesaved(False)

    # If we've set the auto save flag, we'll now save the file
    if Variables.autosave and (".json" in Variables.savedir):
        save()


# ==================================================================#
# Signals the Game Screen to remove one of the chunks
# ==================================================================#
def remove_story_chunk(idx: int):
    emit('from_server', {'cmd': 'removechunk', 'data': idx}, broadcast=True)
    setgamesaved(False)


# ==================================================================#
# Sends the current generator settings to the Game Menu
# ==================================================================#
def refresh_settings():
    # Suppress toggle change events while loading state
    emit('from_server', {'cmd': 'allowtoggle', 'data': False}, broadcast=True)

    if Variables.model != "InferKit":
        emit('from_server', {'cmd': 'updatetemp', 'data': Variables.temp}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopp', 'data': Variables.top_p}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopk', 'data': Variables.top_k}, broadcast=True)
        emit('from_server', {'cmd': 'updatetfs', 'data': Variables.tfs}, broadcast=True)
        emit('from_server', {'cmd': 'updatereppen', 'data': Variables.rep_pen}, broadcast=True)
        emit('from_server', {'cmd': 'updatereppenslope', 'data': Variables.rep_pen_slope}, broadcast=True)
        emit('from_server', {'cmd': 'updatereppenrange', 'data': Variables.rep_pen_range}, broadcast=True)
        emit('from_server', {'cmd': 'updateoutlen', 'data': Variables.genamt}, broadcast=True)
        emit('from_server', {'cmd': 'updatetknmax', 'data': Variables.max_length}, broadcast=True)
        emit('from_server', {'cmd': 'updatenumseq', 'data': Variables.numseqs}, broadcast=True)
    else:
        emit('from_server', {'cmd': 'updatetemp', 'data': Variables.temp}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopp', 'data': Variables.top_p}, broadcast=True)
        emit('from_server', {'cmd': 'updateikgen', 'data': Variables.ikgen}, broadcast=True)

    emit('from_server', {'cmd': 'updateanotedepth', 'data': Variables.andepth}, broadcast=True)
    emit('from_server', {'cmd': 'updatewidepth', 'data': Variables.widepth}, broadcast=True)
    emit('from_server', {'cmd': 'updateuseprompt', 'data': Variables.useprompt}, broadcast=True)
    emit('from_server', {'cmd': 'updateadventure', 'data': Variables.adventure}, broadcast=True)
    emit('from_server', {'cmd': 'updatechatmode', 'data': Variables.chatmode}, broadcast=True)
    emit('from_server', {'cmd': 'updatedynamicscan', 'data': Variables.dynamicscan}, broadcast=True)
    emit('from_server', {'cmd': 'updatenopromptgen', 'data': Variables.nopromptgen}, broadcast=True)
    emit('from_server', {'cmd': 'updaterngpersist', 'data': Variables.rngpersist}, broadcast=True)
    emit('from_server', {'cmd': 'updatenogenmod', 'data': Variables.nogenmod}, broadcast=True)

    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': Variables.formatoptns["frmttriminc"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': Variables.formatoptns["frmtrmblln"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': Variables.formatoptns["frmtrmspch"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': Variables.formatoptns["frmtadsnsp"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatesingleline', 'data': Variables.formatoptns["singleline"]}, broadcast=True)

    # Allow toggle events again
    emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True)


# ==================================================================#
#  Sets the logical and display states for the AI Busy condition
# ==================================================================#
def set_aibusy(state):
    if state:
        Variables.aibusy = True
        emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, broadcast=True)
    else:
        Variables.aibusy = False
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)


# ==================================================================#
# 
# ==================================================================#
def editrequest(n):
    if n == 0:
        txt = Variables.prompt
    else:
        txt = Variables.actions[n - 1]

    Variables.editln = n
    emit('from_server', {'cmd': 'setinputtext', 'data': txt}, broadcast=True)
    emit('from_server', {'cmd': 'enablesubmit', 'data': ''}, broadcast=True)


# ==================================================================#
# 
# ==================================================================#
def editsubmit(data):
    Variables.recentedit = True
    if Variables.editln == 0:
        Variables.prompt = data
    else:
        Variables.actions_metadata[Variables.editln - 1]['Alternative Text'] = \
            Variables.actions_metadata[Variables.editln - 1][
                'Alternative Text'] + [
                {"Text": Variables.actions[Variables.editln - 1],
                 "Pinned": False,
                 "Previous Selection": False,
                 "Edited": True}]
        Variables.actions_metadata[Variables.editln - 1]['Selected Text'] = data
        Variables.actions[Variables.editln - 1] = data

    Variables.mode = "play"
    update_story_chunk(Variables.editln)
    emit('from_server', {'cmd': 'texteffect', 'data': Variables.editln}, broadcast=True)
    emit('from_server', {'cmd': 'editmode', 'data': 'false'})
    send_debug()


# ==================================================================#
#  
# ==================================================================#
def deleterequest():
    Variables.recentedit = True
    # Don't delete prompt
    if Variables.editln == 0:
        # Send error message
        pass
    else:
        Variables.actions_metadata[Variables.editln - 1]['Alternative Text'] = [{"Text": Variables.actions[
            Variables.editln - 1],
                                                                                 "Pinned": False,
                                                                                 "Previous Selection": True,
                                                                                 "Edited": False}] + \
                                                                               Variables.actions_metadata[
                                                                                   Variables.editln - 1][
                                                                                   'Alternative Text']
        Variables.actions_metadata[Variables.editln - 1]['Selected Text'] = ''
        Variables.actions[Variables.editln - 1] = ''
        Variables.mode = "play"
        remove_story_chunk(Variables.editln)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'})
    send_debug()


# ==================================================================#
# 
# ==================================================================#
def inlineedit(chunk, data):
    Variables.recentedit = True
    chunk = int(chunk)
    if chunk == 0:
        if len(data.strip()) == 0:
            return
        Variables.prompt = data
    else:
        if chunk - 1 in Variables.actions:
            Variables.actions_metadata[chunk - 1]['Alternative Text'] = Variables.actions_metadata[chunk - 1][
                                                                            'Alternative Text'] + [
                                                                            {"Text": Variables.actions[chunk - 1],
                                                                             "Pinned": False,
                                                                             "Previous Selection": False,
                                                                             "Edited": True}]
            Variables.actions_metadata[chunk - 1]['Selected Text'] = data
            Variables.actions[chunk - 1] = data
        else:
            print(f"WARNING: Attempted to edit non-existent chunk {chunk}")

    setgamesaved(False)
    update_story_chunk(chunk)
    emit('from_server', {'cmd': 'texteffect', 'data': chunk}, broadcast=True)
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    send_debug()


# ==================================================================#
#  
# ==================================================================#
def inlinedelete(chunk):
    Variables.recentedit = True
    chunk = int(chunk)
    # Don't delete prompt
    if chunk == 0:
        # Send error message
        update_story_chunk(chunk)
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."})
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    else:
        if chunk - 1 in Variables.actions:
            Variables.actions_metadata[chunk - 1]['Alternative Text'] = [{"Text": Variables.actions[chunk - 1],
                                                                          "Pinned": False,
                                                                          "Previous Selection": True,
                                                                          "Edited": False}] + \
                                                                        Variables.actions_metadata[chunk - 1][
                                                                            'Alternative Text']
            Variables.actions_metadata[chunk - 1]['Selected Text'] = ''
            Variables.actions[chunk - 1] = ''
        else:
            print(f"WARNING: Attempted to delete non-existent chunk {chunk}")
        setgamesaved(False)
        remove_story_chunk(chunk)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    send_debug()


# ==================================================================#
#   Toggles the game mode for memory editing and sends UI commands
# ==================================================================#
def togglememorymode():
    if Variables.mode == "play":
        Variables.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True)
        emit('from_server', {'cmd': 'setinputtext', 'data': Variables.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate}, broadcast=True)
    elif Variables.mode == "memory":
        Variables.mode = "play"
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)


# ==================================================================#
#   Toggles the game mode for WI editing and sends UI commands
# ==================================================================#
def togglewimode():
    if Variables.mode == "play":
        Variables.mode = "wi"
        emit('from_server', {'cmd': 'wimode', 'data': 'true'}, broadcast=True)
    elif Variables.mode == "wi":
        # Commit WI fields first
        requestwi()
        # Then set UI state back to Play
        Variables.mode = "play"
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True)
    sendwi()


# ==================================================================#
#   
# ==================================================================#
def addwiitem(folder_uid=None):
    assert folder_uid is None or folder_uid in Variables.wifolders_d
    ob = {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": folder_uid,
          "num": len(Variables.worldinfo),
          "init": False, "selective": False, "constant": False}
    Variables.worldinfo.append(ob)
    while True:
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if uid not in Variables.worldinfo_u:
            break
    Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
    Variables.worldinfo[-1]["uid"] = uid
    if folder_uid is not None:
        Variables.wifolders_u[folder_uid].append(Variables.worldinfo[-1])
    emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True)


# ==================================================================#
#   Creates a new WI folder with an unused cryptographically secure random UID
# ==================================================================#
def addwifolder():
    while True:
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if uid not in Variables.wifolders_d:
            break
    ob = {"name": "", "collapsed": False}
    Variables.wifolders_d[uid] = ob
    Variables.wifolders_l.append(uid)
    Variables.wifolders_u[uid] = []
    emit('from_server', {'cmd': 'addwifolder', 'uid': uid, 'data': ob}, broadcast=True)
    addwiitem(folder_uid=uid)


# ==================================================================#
#   Move the WI entry with UID src so that it immediately precedes
#   the WI entry with UID dst
# ==================================================================#
def movewiitem(dst, src):
    _dst = None
    _src = None
    setgamesaved(False)
    if Variables.worldinfo_u[src]["folder"] is not None:
        for i, e in enumerate(Variables.wifolders_u[Variables.worldinfo_u[src]["folder"]]):
            if e is Variables.worldinfo_u[src]:
                Variables.wifolders_u[Variables.worldinfo_u[src]["folder"]].pop(i)
                break
    if Variables.worldinfo_u[dst]["folder"] is not None:
        Variables.wifolders_u[Variables.worldinfo_u[dst]["folder"]].append(Variables.worldinfo_u[src])
    Variables.worldinfo_u[src]["folder"] = Variables.worldinfo_u[dst]["folder"]
    for i, e in enumerate(Variables.worldinfo):
        if e is Variables.worldinfo_u[src]:
            _src = i
        elif e is Variables.worldinfo_u[dst]:
            _dst = i
    Variables.worldinfo.insert(_dst - (_dst >= _src), Variables.worldinfo.pop(_src))
    sendwi()


# ==================================================================#
#   Move the WI folder with UID src so that it immediately precedes
#   the WI folder with UID dst
# ==================================================================#
def movewifolder(dst, src):
    setgamesaved(False)
    Variables.wifolders_l.remove(src)
    if dst is None:
        # If dst is None, that means we should move src to be the last folder
        Variables.wifolders_l.append(src)
    else:
        Variables.wifolders_l.insert(Variables.wifolders_l.index(dst), src)
    sendwi()


# ==================================================================#
#   
# ==================================================================#
def sendwi():
    # Cache len of WI
    ln = len(Variables.worldinfo)

    # Clear contents of WI container
    emit('from_server',
         {'cmd': 'wistart', 'wifolders_d': Variables.wifolders_d, 'wifolders_l': Variables.wifolders_l, 'data': ''},
         broadcast=True)

    # Stable-sort WI entries in order of folder
    stablesortwi()

    Variables.worldinfo_i = [wi for wi in Variables.worldinfo if wi["init"]]

    # If there are no WI entries, send an empty WI object
    if ln == 0:
        addwiitem()
    else:
        # Send contents of WI array
        last_folder = ...
        for wi in Variables.worldinfo:
            if wi["folder"] != last_folder:
                emit('from_server', {'cmd': 'addwifolder', 'uid': wi["folder"],
                                     'data': Variables.wifolders_d[wi["folder"]] if wi["folder"] is not None else None},
                     broadcast=True)
                last_folder = wi["folder"]
            ob = wi
            emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True)

    emit('from_server', {'cmd': 'wifinish', 'data': ''}, broadcast=True)


# ==================================================================#
#  Request current contents of all WI HTML elements
# ==================================================================#
def requestwi():
    lists = []
    for wi in Variables.worldinfo:
        lists.append(wi["num"])
    emit('from_server', {'cmd': 'requestwiitem', 'data': lists})


# ==================================================================#
#  Stable-sort WI items so that items in the same folder are adjacent,
#  and items in different folders are sorted based on the order of the folders
# ==================================================================#
def stablesortwi():
    mapping = {uid: index for index, uid in enumerate(Variables.wifolders_l)}
    Variables.worldinfo.sort(key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf"))
    last_folder = ...
    last_wi = None
    for i, wi in enumerate(Variables.worldinfo):
        wi["num"] = i
        wi["init"] = True
        if wi["folder"] != last_folder:
            if last_wi is not None and last_folder is not ...:
                last_wi["init"] = False
            last_folder = wi["folder"]
        last_wi = wi
    if last_wi is not None:
        last_wi["init"] = False
    for folder in Variables.wifolders_u:
        Variables.wifolders_u[folder].sort(key=lambda x: x["num"])


# ==================================================================#
#  Extract object from server and send it to WI objects
# ==================================================================#
def commitwi(ar):
    for ob in ar:
        ob["uid"] = int(ob["uid"])
        Variables.worldinfo_u[ob["uid"]]["key"] = ob["key"]
        Variables.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        Variables.worldinfo_u[ob["uid"]]["content"] = ob["content"]
        Variables.worldinfo_u[ob["uid"]]["comment"] = ob.get("comment", "")
        Variables.worldinfo_u[ob["uid"]]["folder"] = ob.get("folder", None)
        Variables.worldinfo_u[ob["uid"]]["selective"] = ob["selective"]
        Variables.worldinfo_u[ob["uid"]]["constant"] = ob.get("constant", False)
    stablesortwi()
    Variables.worldinfo_i = [wi for wi in Variables.worldinfo if wi["init"]]


# ==================================================================#
#  
# ==================================================================#
def deletewi(uid):
    if uid in Variables.worldinfo_u:
        setgamesaved(False)
        # Store UID of deletion request
        Variables.deletewi = uid
        if Variables.deletewi is not None:
            if Variables.worldinfo_u[Variables.deletewi]["folder"] is not None:
                for i, e in enumerate(Variables.wifolders_u[Variables.worldinfo_u[Variables.deletewi]["folder"]]):
                    if e is Variables.worldinfo_u[Variables.deletewi]:
                        Variables.wifolders_u[Variables.worldinfo_u[Variables.deletewi]["folder"]].pop(i)
            for i, e in enumerate(Variables.worldinfo):
                if e is Variables.worldinfo_u[Variables.deletewi]:
                    del Variables.worldinfo[i]
                    break
            del Variables.worldinfo_u[Variables.deletewi]
            # Send the new WI array structure
            sendwi()
            # And reset deletewi
            Variables.deletewi = None


# ==================================================================#
#  
# ==================================================================#
def deletewifolder(uid):
    uid = int(uid)
    del Variables.wifolders_u[uid]
    del Variables.wifolders_d[uid]
    del Variables.wifolders_l[Variables.wifolders_l.index(uid)]
    setgamesaved(False)
    # Delete uninitialized entries in the folder we're going to delete
    Variables.worldinfo = [wi for wi in Variables.worldinfo if wi["folder"] != uid or wi["init"]]
    Variables.worldinfo_i = [wi for wi in Variables.worldinfo if wi["init"]]
    # Move WI entries that are inside of the folder we're going to delete
    # so that they're outside of all folders
    for wi in Variables.worldinfo:
        if wi["folder"] == uid:
            wi["folder"] = None

    sendwi()


# ==================================================================#
#  Look for WI keys in text to generator 
# ==================================================================#
def checkworldinfo(txt, allowed_entries=None, allowed_folders=None, force_use_txt=False, scan_story=True, actions=None):
    original_txt = txt

    if actions is None:
        actions = Variables.actions

    # Dont go any further if WI is empty
    if len(Variables.worldinfo) == 0:
        return "", set()

    # Cache actions length
    ln = len(actions)

    # Don't bother calculating action history if widepth is 0
    if Variables.widepth > 0 and scan_story:
        depth = Variables.widepth
        chunks = None
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if not force_use_txt and (txt != "" and Variables.prompt != txt):
            txt = ""
            depth += 1

        if ln > 0:
            chunks = collections.deque()
            i = 0
            for key in reversed(actions):
                chunk = actions[key]
                chunks.appendleft(chunk)
                i += 1
                if i == depth:
                    break

        if ln >= depth:
            txt = "".join(chunks)
        elif ln > 0:
            txt = Variables.comregex_ai.sub('', Variables.prompt) + "".join(chunks)
        elif ln == 0:
            txt = Variables.comregex_ai.sub('', Variables.prompt)

    if force_use_txt:
        txt += original_txt

    # Scan text for matches on WI keys
    wimem = ""
    found_entries = set()
    for wi in Variables.worldinfo:
        if allowed_entries is not None and wi["uid"] not in allowed_entries:
            continue
        if allowed_folders is not None and wi["folder"] not in allowed_folders:
            continue

        if wi.get("constant", False):
            wimem = wimem + wi["content"] + "\n"
            found_entries.add(id(wi))
            continue

        if (len(wi["key"].strip()) > 0 and (
                not wi.get("selective", False) or len(wi.get("keysecondary", "").strip()) > 0)):
            # Split comma-separated keys
            keys = wi["key"].split(",")
            keys_secondary = wi.get("keysecondary", "").split(",")

            for k in keys:
                ky = k
                # Remove leading/trailing spaces if the option is enabled
                if Variables.wirmvwhtsp:
                    ky = k.strip()
                if ky in txt:
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if Variables.wirmvwhtsp:
                                ksy = ks.strip()
                            if ksy in txt:
                                wimem = wimem + wi["content"] + "\n"
                                found_entries.add(id(wi))
                                found = True
                                break
                        if found:
                            break
                    else:
                        wimem = wimem + wi["content"] + "\n"
                        found_entries.add(id(wi))
                        break

    return wimem, found_entries


# ==================================================================#
#  Commit changes to Memory storage
# ==================================================================#
def memsubmit(data):
    emit('from_server', {'cmd': 'setinputtext', 'data': data}, broadcast=True)
    # Maybe check for length at some point
    # For now just send it to storage
    if data != Variables.memory:
        setgamesaved(False)
    Variables.memory = data
    Variables.mode = "play"
    emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)

    # Ask for contents of Author's Note field
    emit('from_server', {'cmd': 'getanote', 'data': ''})


# ==================================================================#
#  Commit changes to Author's Note
# ==================================================================#
def anotesubmit(data, template=""):
    assert type(data) is str and type(template) is str
    # Maybe check for length at some point
    # For now just send it to storage
    if data != Variables.authornote:
        setgamesaved(False)
    Variables.authornote = data

    if Variables.authornotetemplate != template:
        Variables.setauthornotetemplate = template
        settingschanged()
    Variables.authornotetemplate = template

    emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote}, broadcast=True)
    emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate}, broadcast=True)


# ==================================================================#
#  Assembles game data into a request to InferKit API
# ==================================================================#
def ikrequest(txt):
    # Log request to console
    if not Variables.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(Colors.YELLOW, len(txt), txt, Colors.END))

    # Build request JSON data
    reqdata = {
        'forceNoEnd': True,
        'length': Variables.ikgen,
        'prompt': {
            'isContinuation': False,
            'text': txt
        },
        'startFromBeginning': False,
        'streamResponse': False,
        'temperature': Variables.temp,
        'topP': Variables.top_p
    }

    # Create request
    req = requests.post(
        Variables.url,
        json=reqdata,
        headers={
            'Authorization': 'Bearer ' + Variables.apikey
        }
    )

    # Deal with the response
    if req.status_code == 200:
        genout = req.json()["data"]["text"]

        Variables.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if Variables.lua_koboldbridge.regeneration_required:
            Variables.lua_koboldbridge.regeneration_required = False
            genout = Variables.lua_koboldbridge.outputs[1]
            assert genout is str

        if not Variables.quiet:
            print("{0}{1}{2}".format(Colors.CYAN, genout, Colors.END))
        Variables.actions.append(genout)
        if Variables.actions.get_last_key() in Variables.actions_metadata:
            Variables.actions_metadata[Variables.actions.get_last_key()] = {"Selected Text": genout,
                                                                            "Alternative Text": []}
        else:
            # 2. We've selected a chunk of text that is was presented previously
            alternatives = [item['Text'] for item in
                            Variables.actions_metadata[Variables.actions.get_last_key()]["Alternative Text"]]
            if genout in alternatives:
                alternatives = [item for item in
                                Variables.actions_metadata[Variables.actions.get_last_key()]["Alternative Text"]
                                if item['Text'] != genout]
                Variables.actions_metadata[Variables.actions.get_last_key()]["Alternative Text"] = alternatives
            Variables.actions_metadata[Variables.actions.get_last_key()]["Selected Text"] = genout
        update_story_chunk('last')
        emit('from_server',
             {'cmd': 'texteffect', 'data': Variables.actions.get_last_key() + 1 if len(Variables.actions) else 0},
             broadcast=True)
        send_debug()
        set_aibusy(0)
    else:
        # Send error message to web client
        code = "null"
        er = req.json()
        if "error" in er:
            code = er["error"]["extensions"]["code"]
        elif "errors" in er:
            code = er["errors"][0]["extensions"]["code"]

        errmsg = "InferKit API Error: {0} - {1}".format(req.status_code, code)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)


# ==================================================================#
#  Assembles game data into a request to OpenAI API
# ==================================================================#
def oairequest(txt, maxi):
    # Log request to console
    if not Variables.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(Colors.YELLOW, len(txt), txt, Colors.END))

    # Store context in memory to use it for comparison with generated content
    Variables.lastctx = txt

    # Build request JSON data
    reqdata = {
        'prompt': txt,
        'max_tokens': maxi,
        'temperature': Variables.temp,
        'top_p': Variables.top_p,
        'top_k': Variables.top_k,
        'tfs': Variables.tfs,
        'repetition_penalty': Variables.rep_pen,
        'repetition_penalty_slope': Variables.rep_pen_slope,
        'repetition_penalty_range': Variables.rep_pen_range,
        'n': 1,
        'stream': False
    }

    req = requests.post(
        Variables.oaiurl,
        json=reqdata,
        headers={
            'Authorization': 'Bearer ' + Variables.oaiapikey,
            'Content-Type': 'application/json'
        }
    )

    # Deal with the response
    if req.status_code == 200:
        genout = req.json()["choices"][0]["text"]

        Variables.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if Variables.lua_koboldbridge.regeneration_required:
            Variables.lua_koboldbridge.regeneration_required = False
            genout = Variables.lua_koboldbridge.outputs[1]
            assert genout is str

        if not Variables.quiet:
            print("{0}{1}{2}".format(Colors.CYAN, genout, Colors.END))
        Variables.actions.append(genout)
        if Variables.actions.get_last_key() in Variables.actions_metadata:
            Variables.actions_metadata[Variables.actions.get_last_key()] = {"Selected Text": genout,
                                                                            "Alternative Text": []}
        else:
            # 2. We've selected a chunk of text that is was presented previously
            alternatives = [item['Text'] for item in
                            Variables.actions_metadata[Variables.actions.get_last_key()]["Alternative Text"]]
            if genout in alternatives:
                alternatives = [item for item in
                                Variables.actions_metadata[Variables.actions.get_last_key()]["Alternative Text"]
                                if item['Text'] != genout]
                Variables.actions_metadata[Variables.actions.get_last_key()]["Alternative Text"] = alternatives
            Variables.actions_metadata[Variables.actions.get_last_key()]["Selected Text"] = genout
        update_story_chunk('last')
        emit('from_server',
             {'cmd': 'texteffect', 'data': Variables.actions.get_last_key() + 1 if len(Variables.actions) else 0},
             broadcast=True)
        send_debug()
        set_aibusy(0)
    else:
        err_type = "null"
        message = "null"
        # Send error message to web client            
        er = req.json()
        if "error" in er:
            err_type = er["error"]["type"]
            message = er["error"]["message"]

        errmsg = "OpenAI API Error: {0} - {1}".format(err_type, message)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)


# ==================================================================#
#  Forces UI to Play mode
# ==================================================================#
def exitmodes():
    if Variables.mode == "edit":
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    elif Variables.mode == "memory":
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)
    elif Variables.mode == "wi":
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True)
    Variables.mode = "play"


# ==================================================================#
#  Launch in-browser save prompt
# ==================================================================#
def saveas(data):
    name = data['name']
    savepins = data['pins']
    # Check if filename exists already
    name = utils.cleanfilename(name)
    if not fileops.saveexists(name) or (Variables.saveow and Variables.svowname == name):
        # All clear to save
        e = saverequest(fileops.storypath(name), savepins=savepins)
        Variables.saveow = False
        Variables.svowname = ""
        if e is None:
            emit('from_server', {'cmd': 'hidesaveas', 'data': ''})
        else:
            print("{0}{1}{2}".format(Colors.RED, str(e), Colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)})
    else:
        # File exists, prompt for overwrite
        Variables.saveow = True
        Variables.svowname = name
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''})


# ==================================================================#
#  Launch in-browser story-delete prompt
# ==================================================================#
def deletesave(name):
    name = utils.cleanfilename(name)
    e = fileops.deletesave(name)
    if e is None:
        if Variables.smandelete:
            emit('from_server', {'cmd': 'hidepopupdelete', 'data': ''})
            getloadlist()
        else:
            emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to delete this story"})
    else:
        print("{0}{1}{2}".format(Colors.RED, str(e), Colors.END))
        emit('from_server', {'cmd': 'popuperror', 'data': str(e)})


# ==================================================================#
#  Launch in-browser story-rename prompt
# ==================================================================#
def renamesave(name, newname):
    # Check if filename exists already
    name = utils.cleanfilename(name)
    newname = utils.cleanfilename(newname)
    if not fileops.saveexists(newname) or name == newname or (Variables.saveow and Variables.svowname == newname):
        e = fileops.renamesave(name, newname)
        Variables.saveow = False
        Variables.svowname = ""
        if e is None:
            if Variables.smanrename:
                emit('from_server', {'cmd': 'hidepopuprename', 'data': ''})
                getloadlist()
            else:
                emit('from_server',
                     {'cmd': 'popuperror', 'data': "The server denied your request to rename this story"})
        else:
            print("{0}{1}{2}".format(Colors.RED, str(e), Colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)})
    else:
        # File exists, prompt for overwrite
        Variables.saveow = True
        Variables.svowname = newname
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''})


# ==================================================================#
#  Save the currently running story
# ==================================================================#
def save():
    # Check if a file is currently open
    if ".json" in Variables.savedir:
        saverequest(Variables.savedir)
    else:
        emit('from_server', {'cmd': 'saveas', 'data': ''})


# ==================================================================#
#  Save the story via file browser
# ==================================================================#
def savetofile():
    savpath = fileops.getsavepath(Variables.savedir, "Save Story As", [("Json", "*.json")])
    saverequest(savpath)


# ==================================================================#
#  Save the story to specified path
# ==================================================================#
def saverequest(savpath, savepins=True):
    if savpath:
        # Leave Edit/Memory mode before continuing
        exitmodes()

        # Save path for future saves
        Variables.savedir = savpath
        txtpath = os.path.splitext(savpath)[0] + ".txt"
        # Build json to write
        js = {}
        js["gamestarted"] = Variables.gamestarted
        js["prompt"] = Variables.prompt
        js["memory"] = Variables.memory
        js["authorsnote"] = Variables.authornote
        js["anotetemplate"] = Variables.authornotetemplate
        js["actions"] = tuple(Variables.actions.values())
        if savepins:
            js["actions_metadata"] = Variables.actions_metadata
        js["worldinfo"] = []
        js["wifolders_d"] = Variables.wifolders_d
        js["wifolders_l"] = Variables.wifolders_l

        # Extract only the important bits of WI
        for wi in Variables.worldinfo_i:
            if True:
                js["worldinfo"].append({
                    "key": wi["key"],
                    "keysecondary": wi["keysecondary"],
                    "content": wi["content"],
                    "comment": wi["comment"],
                    "folder": wi["folder"],
                    "selective": wi["selective"],
                    "constant": wi["constant"]
                })

        txt = Variables.prompt + "".join(Variables.actions.values())

        # Write it
        try:
            file = open(savpath, "w")
        except Exception as e:
            return e
        try:
            file.write(json.dumps(js, indent=3))
        except Exception as e:
            file.close()
            return e
        file.close()

        try:
            file = open(txtpath, "w")
        except Exception as e:
            return e
        try:
            file.write(txt)
        except Exception as e:
            file.close()
            return e
        file.close()

        filename = path.basename(savpath)
        if filename.endswith('.json'):
            filename = filename[:-5]
        Variables.laststory = filename
        emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory}, broadcast=True)
        setgamesaved(True)
        print("{0}Story saved to {1}!{2}".format(Colors.GREEN, path.basename(savpath), Colors.END))


# ==================================================================#
#  Show list of saved stories
# ==================================================================#
def getloadlist():
    emit('from_server', {'cmd': 'buildload', 'data': fileops.getstoryfiles()})


# ==================================================================#
#  Show list of soft prompts
# ==================================================================#
def getsplist():
    if Variables.allowsp:
        emit('from_server', {'cmd': 'buildsp', 'data': fileops.getspfiles(Variables.modeldim)})


# ==================================================================#
#  Get list of userscripts
# ==================================================================#
def getuslist():
    files = {i: v for i, v in enumerate(fileops.getusfiles())}
    loaded = []
    unloaded = []
    userscripts = set(Variables.userscripts)
    for i in range(len(files)):
        if files[i]["filename"] not in userscripts:
            unloaded.append(files[i])
    files = {files[k]["filename"]: files[k] for k in files}
    userscripts = set(files.keys())
    for filename in Variables.userscripts:
        if filename in userscripts:
            loaded.append(files[filename])
    return unloaded, loaded


# ==================================================================#
#  Load a saved story via file browser
# ==================================================================#
def loadfromfile():
    loadpath = fileops.getloadpath(Variables.savedir, "Select Story File", [("Json", "*.json")])
    loadrequest(loadpath)


# ==================================================================#
#  Load a stored story from a file
# ==================================================================#
def loadrequest(loadpath, filename=None):
    if loadpath:
        # Leave Edit/Memory mode before continuing
        exitmodes()

        # Read file contents into JSON object
        if isinstance(loadpath, str):
            with open(loadpath, "r") as file:
                js = json.load(file)
            if filename is None:
                filename = path.basename(loadpath)
        else:
            js = loadpath
            if filename is None:
                filename = "untitled.json"

        # Copy file contents to vars
        Variables.gamestarted = js["gamestarted"]
        Variables.prompt = js["prompt"]
        Variables.memory = js["memory"]
        Variables.worldinfo = []
        Variables.worldinfo = []
        Variables.worldinfo_u = {}
        Variables.wifolders_d = {int(k): v for k, v in js.get("wifolders_d", {}).items()}
        Variables.wifolders_l = js.get("wifolders_l", [])
        Variables.wifolders_u = {uid: [] for uid in Variables.wifolders_d}
        Variables.lastact = ""
        Variables.submission = ""
        Variables.lastctx = ""

        del Variables.actions
        Variables.actions = structures.KoboldStoryRegister()
        actions = collections.deque(js["actions"])

        if "actions_metadata" in js:

            if type(js["actions_metadata"]) == dict:
                temp = js["actions_metadata"]
                Variables.actions_metadata = {}
                # we need to redo the numbering of the actions_metadata since the actions list doesn't preserve it's number on saving
                if len(temp) > 0:
                    counter = 0
                    temp = {int(k): v for k, v in temp.items()}
                    for i in range(max(temp) + 1):
                        if i in temp:
                            Variables.actions_metadata[counter] = temp[i]
                            counter += 1
                del temp
            else:
                # fix if we're using the old metadata format
                Variables.actions_metadata = {}
                i = 0

                for text in js['actions']:
                    Variables.actions_metadata[i] = {'Selected Text': text, 'Alternative Text': []}
                    i += 1
        else:
            Variables.actions_metadata = {}
            i = 0

            for text in js['actions']:
                Variables.actions_metadata[i] = {'Selected Text': text, 'Alternative Text': []}
                i += 1

        if len(Variables.prompt.strip()) == 0:
            while len(actions):
                action = actions.popleft()
                if len(action.strip()) != 0:
                    Variables.prompt = action
                    break
            else:
                Variables.gamestarted = False
        if Variables.gamestarted:
            for s in actions:
                Variables.actions.append(s)

        # Try not to break older save files
        if "authorsnote" in js:
            Variables.authornote = js["authorsnote"]
        else:
            Variables.authornote = ""
        if "anotetemplate" in js:
            Variables.authornotetemplate = js["anotetemplate"]
        else:
            Variables.authornotetemplate = "[Author's note: <|>]"

        if "worldinfo" in js:
            num = 0
            for wi in js["worldinfo"]:
                Variables.worldinfo.append({
                    "key": wi["key"],
                    "keysecondary": wi.get("keysecondary", ""),
                    "content": wi["content"],
                    "comment": wi.get("comment", ""),
                    "folder": wi.get("folder", None),
                    "num": num,
                    "init": True,
                    "selective": wi.get("selective", False),
                    "constant": wi.get("constant", False),
                    "uid": None,
                })
                while True:
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if uid not in Variables.worldinfo_u:
                        break
                Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
                Variables.worldinfo[-1]["uid"] = uid
                if Variables.worldinfo[-1]["folder"] is not None:
                    Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
                num += 1

        for uid in Variables.wifolders_l + [None]:
            Variables.worldinfo.append(
                {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False,
                 "selective": False, "constant": False, "uid": None})
            while True:
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if uid not in Variables.worldinfo_u:
                    break
            Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
            Variables.worldinfo[-1]["uid"] = uid
            if Variables.worldinfo[-1]["folder"] is not None:
                Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
        stablesortwi()
        Variables.worldinfo_i = [wi for wi in Variables.worldinfo if wi["init"]]

        # Save path for save button
        Variables.savedir = loadpath

        # Clear loadselect var
        Variables.loadselect = ""

        # Refresh game screen
        _filename = filename
        if filename.endswith('.json'):
            _filename = filename[:-5]
        Variables.laststory = _filename
        emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory}, broadcast=True)
        setgamesaved(True)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
        print("{0}Story loaded from {1}!{2}".format(Colors.GREEN, filename, Colors.END))

        send_debug()


# ==================================================================#
# Import an AIDungon game exported with Mimi's tool
# ==================================================================#
def importrequest():
    importpath = fileops.getloadpath(Variables.savedir, "Select AID CAT File", [("Json", "*.json")])

    if importpath:
        # Leave Edit/Memory mode before continuing
        exitmodes()

        # Read file contents into JSON object
        file = open(importpath, "rb")
        Variables.importjs = json.load(file)

        # If a bundle file is being imported, select just the Adventures object
        if type(Variables.importjs) is dict and "stories" in Variables.importjs:
            Variables.importjs = Variables.importjs["stories"]

        # Clear Popup Contents
        emit('from_server', {'cmd': 'clearpopup', 'data': ''}, broadcast=True)

        # Initialize vars
        num = 0
        Variables.importnum = -1

        # Get list of stories
        for story in Variables.importjs:
            ob = {}
            ob["num"] = num
            if story["title"] != "" and story["title"] is not None:
                ob["title"] = story["title"]
            else:
                ob["title"] = "(No Title)"
            if story["description"] != "" and story["description"] is not None:
                ob["descr"] = story["description"]
            else:
                ob["descr"] = "(No Description)"
            if "actions" in story:
                ob["acts"] = len(story["actions"])
            elif "actionWindow" in story:
                ob["acts"] = len(story["actionWindow"])
            emit('from_server', {'cmd': 'addimportline', 'data': ob})
            num += 1

        # Show Popup
        emit('from_server', {'cmd': 'popupshow', 'data': True})


# ==================================================================#
# Import an AIDungon game selected in popup
# ==================================================================#
def importgame():
    if Variables.importnum >= 0:
        # Cache reference to selected game
        ref = Variables.importjs[Variables.importnum]

        # Copy game contents to vars
        Variables.gamestarted = True

        # Support for different versions of export script
        if "actions" in ref:
            if len(ref["actions"]) > 0:
                Variables.prompt = ref["actions"][0]["text"]
            else:
                Variables.prompt = ""
        elif "actionWindow" in ref:
            if len(ref["actionWindow"]) > 0:
                Variables.prompt = ref["actionWindow"][0]["text"]
            else:
                Variables.prompt = ""
        else:
            Variables.prompt = ""
        Variables.memory = ref["memory"]
        Variables.authornote = ref["authorsNote"] if type(ref["authorsNote"]) is str else ""
        Variables.authornotetemplate = "[Author's note: <|>]"
        Variables.actions = structures.KoboldStoryRegister()
        Variables.actions_metadata = {}
        Variables.worldinfo = []
        Variables.worldinfo_i = []
        Variables.worldinfo_u = {}
        Variables.wifolders_d = {}
        Variables.wifolders_l = []
        Variables.wifolders_u = {uid: [] for uid in Variables.wifolders_d}
        Variables.lastact = ""
        Variables.submission = ""
        Variables.lastctx = ""

        # Get all actions except for prompt
        if "actions" in ref:
            if len(ref["actions"]) > 1:
                for act in ref["actions"][1:]:
                    Variables.actions.append(act["text"])
        elif "actionWindow" in ref:
            if len(ref["actionWindow"]) > 1:
                for act in ref["actionWindow"][1:]:
                    Variables.actions.append(act["text"])

        # Get just the important parts of world info
        if ref["worldInfo"] is not None:
            if len(ref["worldInfo"]) > 1:
                num = 0
                for wi in ref["worldInfo"]:
                    Variables.worldinfo.append({
                        "key": wi["keys"],
                        "keysecondary": wi.get("keysecondary", ""),
                        "content": wi["entry"],
                        "comment": wi.get("comment", ""),
                        "folder": wi.get("folder", None),
                        "num": num,
                        "init": True,
                        "selective": wi.get("selective", False),
                        "constant": wi.get("constant", False),
                        "uid": None,
                    })
                    while True:
                        uid = int.from_bytes(os.urandom(4), "little", signed=True)
                        if uid not in Variables.worldinfo_u:
                            break
                    Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
                    Variables.worldinfo[-1]["uid"] = uid
                    if (Variables.worldinfo[-1]["folder"]) is not None:
                        Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
                    num += 1

        for uid in Variables.wifolders_l + [None]:
            Variables.worldinfo.append(
                {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False,
                 "selective": False, "constant": False, "uid": None})
            while True:
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if uid not in Variables.worldinfo_u:
                    break
            Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
            Variables.worldinfo[-1]["uid"] = uid
            if Variables.worldinfo[-1]["folder"] is not None:
                Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
        stablesortwi()
        Variables.worldinfo_i = [wi for wi in Variables.worldinfo if wi["init"]]

        # Clear import data
        Variables.importjs = {}

        # Reset current save
        Variables.savedir = getcwd() + "\\stories"

        # Refresh game screen
        Variables.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory}, broadcast=True)
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)


# ==================================================================#
# Import an aidg.club prompt and start a new game with it.
# ==================================================================#
def importaidgrequest(aidg_id):
    exitmodes()

    urlformat = "https://prompts.aidg.club/api/"
    req = requests.get(urlformat + aidg_id)

    if req.status_code == 200:
        js = req.json()

        # Import game state
        Variables.gamestarted = True
        Variables.prompt = js["promptContent"]
        Variables.memory = js["memory"]
        Variables.authornote = js["authorsNote"]
        Variables.authornotetemplate = "[Author's note: <|>]"
        Variables.actions = structures.KoboldStoryRegister()
        Variables.actions_metadata = {}
        Variables.worldinfo = []
        Variables.worldinfo_i = []
        Variables.worldinfo_u = {}
        Variables.wifolders_d = {}
        Variables.wifolders_l = []
        Variables.wifolders_u = {uid: [] for uid in Variables.wifolders_d}
        Variables.lastact = ""
        Variables.submission = ""
        Variables.lastctx = ""

        num = 0
        for wi in js["worldInfos"]:
            Variables.worldinfo.append({
                "key": wi["keys"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["entry"],
                "comment": wi.get("comment", ""),
                "folder": wi.get("folder", None),
                "num": num,
                "init": True,
                "selective": wi.get("selective", False),
                "constant": wi.get("constant", False),
                "uid": None,
            })
            while True:
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if uid not in Variables.worldinfo_u:
                    break
            Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
            Variables.worldinfo[-1]["uid"] = uid
            if (Variables.worldinfo[-1]["folder"]) is not None:
                Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
            num += 1

        for uid in Variables.wifolders_l + [None]:
            Variables.worldinfo.append(
                {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False,
                 "selective": False, "constant": False, "uid": None})
            while True:
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if uid not in Variables.worldinfo_u:
                    break
            Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
            Variables.worldinfo[-1]["uid"] = uid
            if Variables.worldinfo[-1]["folder"] is not None:
                Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
        stablesortwi()
        Variables.worldinfo_i = [wi for wi in Variables.worldinfo if wi["init"]]

        # Reset current save
        Variables.savedir = getcwd() + "\\stories"

        # Refresh game screen
        Variables.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory}, broadcast=True)
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)


# ==================================================================#
#  Import World Info JSON file
# ==================================================================#
def wiimportrequest():
    importpath = fileops.getloadpath(Variables.savedir, "Select World Info File", [("Json", "*.json")])
    if importpath:
        file = open(importpath, "rb")
        js = json.load(file)
        if len(js) > 0:
            # If the most recent WI entry is blank, remove it.
            if not Variables.worldinfo[-1]["init"]:
                del Variables.worldinfo[-1]
            # Now grab the new stuff
            num = len(Variables.worldinfo)
            for wi in js:
                Variables.worldinfo.append({
                    "key": wi["keys"],
                    "keysecondary": wi.get("keysecondary", ""),
                    "content": wi["entry"],
                    "comment": wi.get("comment", ""),
                    "folder": wi.get("folder", None),
                    "num": num,
                    "init": True,
                    "selective": wi.get("selective", False),
                    "constant": wi.get("constant", False),
                    "uid": None,
                })
                while True:
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if uid not in Variables.worldinfo_u:
                        break
                Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
                Variables.worldinfo[-1]["uid"] = uid
                if (Variables.worldinfo[-1]["folder"]) is not None:
                    Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])
                num += 1
            for uid in [None]:
                Variables.worldinfo.append(
                    {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None,
                     "init": False, "selective": False, "constant": False, "uid": None})
                while True:
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if uid not in Variables.worldinfo_u:
                        break
                Variables.worldinfo_u[uid] = Variables.worldinfo[-1]
                Variables.worldinfo[-1]["uid"] = uid
                if Variables.worldinfo[-1]["folder"] is not None:
                    Variables.wifolders_u[Variables.worldinfo[-1]["folder"]].append(Variables.worldinfo[-1])

        if not Variables.quiet:
            print("{0}".format(Variables.worldinfo[0]))

        # Refresh game screen
        setgamesaved(False)
        sendwi()


# ==================================================================#
#  Starts a new story
# ==================================================================#
def newgamerequest():
    # Leave Edit/Memory mode before continuing
    exitmodes()

    # Clear vars values
    Variables.gamestarted = False
    Variables.prompt = ""
    Variables.memory = ""
    Variables.actions = structures.KoboldStoryRegister()
    Variables.actions_metadata = {}

    Variables.authornote = ""
    Variables.authornotetemplate = Variables.setauthornotetemplate
    Variables.worldinfo = []
    Variables.worldinfo_i = []
    Variables.worldinfo_u = {}
    Variables.wifolders_d = {}
    Variables.wifolders_l = []
    Variables.lastact = ""
    Variables.submission = ""
    Variables.lastctx = ""

    # Reset current save
    Variables.savedir = getcwd() + "\\stories"

    # Refresh game screen
    Variables.laststory = None
    emit('from_server', {'cmd': 'setstoryname', 'data': Variables.laststory}, broadcast=True)
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory}, broadcast=True)
    emit('from_server', {'cmd': 'setanote', 'data': Variables.authornote}, broadcast=True)
    emit('from_server', {'cmd': 'setanotetemplate', 'data': Variables.authornotetemplate}, broadcast=True)
    setstartstate()


def randomgamerequest(topic, memory=""):
    if Variables.noai:
        newgamerequest()
        Variables.memory = memory
        emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory}, broadcast=True)
        return
    Variables.recentrng = topic
    Variables.recentrngm = memory
    newgamerequest()
    setgamesaved(False)
    _memory = memory
    if len(memory) > 0:
        _memory = memory.rstrip() + "\n\n"
    Variables.memory = _memory + "You generate the following " + topic + " story concept :"
    Variables.lua_koboldbridge.feedback = None
    actionsubmit("", force_submit=True, force_prompt_gen=True)
    Variables.memory = memory
    emit('from_server', {'cmd': 'setmemory', 'data': Variables.memory}, broadcast=True)


# Prevent tokenizer from taking extra time the first time it's used
def __preempt_tokenizer():
    if "tokenizer" not in globals():
        return
    utils.decodenewlines(tokenizer.decode([25678, 559]), Variables.newlinemode)
    tokenizer.encode(utils.encodenewlines("eunoia", Variables.newlinemode))


threading.Thread(target=__preempt_tokenizer).start()

# Load soft prompt specified by the settings file, if applicable
if path.exists("settings/" + getmodelname().replace('/', '_') + ".settings"):
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
    js = json.load(file)
    if (Variables.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(
            q not in js["softprompt"] for q in ("..", ":")) and (
            len(js["softprompt"]) == 0 or all(js["softprompt"][0] not in q for q in ("/", "\\")))):
        sprequest(js["softprompt"])
    else:
        Variables.spfilename = ""
    file.close()

# Precompile TPU backend if required
if Variables.use_colab_tpu or Variables.model in ("TPUMeshTransformerGPTJ",):
    soft_tokens = tpumtjgetsofttokens()
    if Variables.dynamicscan or (not Variables.nogenmod and Variables.has_genmod):
        threading.Thread(
            target=tpu_mtj_backend.infer_dynamic,
            args=(np.tile(np.uint32((23403, 727, 20185)), (Variables.numseqs, 1)),),
            kwargs={
                "soft_embeddings": Variables.sp,
                "soft_tokens": soft_tokens,
                "gen_len": 1,
                "use_callback": False,
                "numseqs": Variables.numseqs,
                "excluded_world_info": list(set() for _ in range(Variables.numseqs)),
            },
        ).start()
    else:
        threading.Thread(
            target=tpu_mtj_backend.infer_static,
            args=(np.uint32((23403, 727, 20185)),),
            kwargs={
                "soft_embeddings": Variables.sp,
                "soft_tokens": soft_tokens,
                "gen_len": 1,
                "numseqs": Variables.numseqs,
            },
        ).start()


def send_debug():
    if Variables.debug:
        debug_info = ""
        try:
            debug_info = "{}Newline Mode: {}\n".format(debug_info, Variables.newlinemode)
        except:
            pass
        try:
            debug_info = "{}Action Length: {}\n".format(debug_info, Variables.actions.get_last_key())
        except:
            pass
        try:
            debug_info = "{}Actions Metadata Length: {}\n".format(debug_info, max(Variables.actions_metadata) if len(
                Variables.actions_metadata) > 0 else 0)
        except:
            pass
        try:
            debug_info = "{}Actions: {}\n".format(debug_info, [k for k in Variables.actions])
        except:
            pass
        try:
            debug_info = "{}Actions Metadata: {}\n".format(debug_info, [k for k in Variables.actions_metadata])
        except:
            pass
        try:
            debug_info = "{}Last Action: {}\n".format(debug_info, Variables.actions[Variables.actions.get_last_key()])
        except:
            pass
        try:
            debug_info = "{}Last Metadata: {}\n".format(debug_info,
                                                        Variables.actions_metadata[max(Variables.actions_metadata)])
        except:
            pass

        emit('from_server', {'cmd': 'debug_info', 'data': debug_info}, broadcast=True)


# ==================================================================#
#  Final startup commands to launch Flask app
# ==================================================================#
print("", end="", flush=True)
if __name__ == "__main__":
    print("{0}\nStarting webserver...{1}".format(Colors.GREEN, Colors.END), flush=True)

    # Start Flask/SocketIO (Blocking, so this must be last method!)

    # socketio.run(app, host='0.0.0.0', port=5000)
    cloudflare = "undefined"
    if Variables.host:
        if args.ngrok:
            from flask_ngrok import _run_ngrok

            cloudflare = _run_ngrok()
        elif args.remote:
            from flask_cloudflared import _run_cloudflared

            cloudflare = _run_cloudflared(5000)
        if args.ngrok or args.remote:
            with open('cloudflare.log', 'w') as cloudflarelog:
                cloudflarelog.write(
                    "KoboldAI has finished loading and is available at the following link : " + cloudflare)
                print(format(
                    Colors.GREEN) + "KoboldAI has finished loading and is available at the following link : "
                      + cloudflare + format(
                    Colors.END))
        else:
            print("{0}Webserver has started, you can now connect to this machine at port 5000{1}".format(Colors.GREEN,
                                                                                                         Colors.END))
        Variables.serverstarted = True
        socketio.run(app, host='0.0.0.0', port=5000)
    else:
        import webbrowser

        webbrowser.open_new('http://localhost:5000')
        print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:5000/{1}".format(Colors.GREEN,
                                                                                                           Colors.END))
        Variables.serverstarted = True
        if args.unblock:
            socketio.run(app, port=5000, host='0.0.0.0')
        else:
            socketio.run(app, port=5000)

else:
    print("{0}\nServer started in WSGI mode!{1}".format(Colors.GREEN, Colors.END), flush=True)
