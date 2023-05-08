#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-remote/bin/python" ]; then
  ./install_requirements.sh remote
fi
bin/micromamba run -r runtime -n koboldai python aiserver.py $*
