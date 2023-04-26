#!/bin/bash
export PYTHONNOUSERSITE=1
if [ ! -f "runtime/envs/koboldai-oneapi/bin/python" ]; then
./install_requirements.sh oneapi
fi
source /opt/intel/oneapi/setvars.sh
export IPEX=1
bin/micromamba run -r runtime -n koboldai-oneapi python aiserver.py --ipex $*
