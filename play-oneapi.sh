#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-oneapi/bin/python" ]; then
pip install https://github.com/TotalDay/Intel_ARC_GPU_WSL_Stable_Diffusion_WEBUI/releases/download/intel_extension_for_pytorch/torch-1.13.0a0+git7c98e70-cp310-cp310-linux_x86_64.whl
pip install https://github.com/TotalDay/Intel_ARC_GPU_WSL_Stable_Diffusion_WEBUI/releases/download/intel_extension_for_pytorch/intel_extension_for_pytorch-1.13.10+xpu-cp310-cp310-linux_x86_64.whl
./install_requirements.sh oneapi
fi
source /opt/intel/oneapi/setvars.sh
export IPEX=1
bin/micromamba run -r runtime -n koboldai-oneapi python aiserver.py --ipex $*
