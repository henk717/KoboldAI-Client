#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-oneapi/bin/python" ]; then
./install_requirements.sh oneapi
fi
source /opt/intel/oneapi/setvars.sh
bin/micromamba run -r runtime -n koboldai-oneapi python aiserver.py --use-ipex --use-channels-last $*
