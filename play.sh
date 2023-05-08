#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-cuda/bin/python" ]; then
  ./install_requirements.sh cuda
fi
bin/micromamba run -r runtime -n koboldai-cuda python aiserver.py $*
