#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai/bin/python" ]; then
./install_requirements.sh cuda
fi
bin/micromamba run -r runtime -p runtime/envs/koboldai python aiserver.py $*
