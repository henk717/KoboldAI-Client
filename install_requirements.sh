#!/bin/bash
export PYTHONNOUSERSITE=1
git submodule update --init --recursive
if [[ $1 = "cuda" || $1 = "CUDA" ]]; then
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environments/huggingface.yml -r runtime -n koboldai -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/huggingface.yml -r runtime -n koboldai -y
exit
fi
if [[ $1 = "rocm" || $1 = "ROCM" ]]; then
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environments/rocm.yml -r runtime -n koboldai-rocm -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/rocm.yml -r runtime -n koboldai-rocm -y
exit
fi
if [[ $1 = "oneapi" || $1 = "ONEAPI" || $1 = "OneAPI" || $1 = "oneAPI" ]]; then
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environments/oneapi.yml -r runtime -n koboldai-oneapi -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/oneapi.yml -r runtime -n koboldai-oneapi -y
#Wants torch>=1.13.0 but can't find it even tough we have it.
bin/micromamba run -r runtime -n koboldai-oneapi pip install --no-deps peft==0.3.0 accelerate==0.18.0
exit
fi
echo Please specify either CUDA or ROCM or oneAPI
