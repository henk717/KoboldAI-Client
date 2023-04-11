cd docker-oneapi
xhost +local:docker
cp ../environments/oneapi.yml env.yml
docker-compose run --service-ports koboldai bash -c "cd /content && source /opt/intel/oneapi/setvars.sh && export IPEX=1 && python3 aiserver.py --ipex $*"
