#!/bin/bash
export PYTHONNOUSERSITE=1
git submodule update --init --recursive
if [[ $1 = "cuda" || $1 = "CUDA" ]]; then
wget --no-iri -qO- https://github.com/mamba-org/micromamba-releases/releases/download/1.5.8-0/micromamba-linux-64.tar.bz2 | tar -xvj bin/micromamba
bin/micromamba create -f environments/huggingface.yml -r runtime -p runtime/envs/koboldai -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/huggingface.yml -r runtime -p runtime/envs/koboldai -y
exit
fi
if [[ $1 = "rocm" || $1 = "ROCM" ]]; then
wget --no-iri -qO- https://github.com/mamba-org/micromamba-releases/releases/download/1.5.8-0/micromamba-linux-64.tar.bz2 | tar -xvj bin/micromamba
bin/micromamba create -f environments/rocm.yml -r runtime -p runtime/envs/koboldai-rocm -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/rocm.yml -r runtime -p runtime/envs/koboldai-rocm -y
exit
fi
if [[ $1 = "ipex" || $1 = "IPEX" ]]; then
wget --no-iri -qO- https://github.com/mamba-org/micromamba-releases/releases/download/1.5.8-0/micromamba-linux-64.tar.bz2 | tar -xvj bin/micromamba
bin/micromamba create -f environments/ipex.yml -r runtime -p runtime/envs/koboldai-ipex -y
# Weird micromamba bug causes it to fail the first time, running it twice just to be safe, the second time is much faster
bin/micromamba create -f environments/ipex.yml -r runtime -p runtime/envs/koboldai-ipex -y
exit
fi
echo Please specify either CUDA or ROCM or IPEX
