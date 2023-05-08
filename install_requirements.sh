#!/bin/bash
export PYTHONNOUSERSITE=1
git submodule update --init --recursive

platform_name() {
  uname="$(uname -a | tr "[:upper:]" "[:lower:]")"
  case "$uname" in
    *darwin*arm64*) echo osx-arm64 ;;
    *darwin*x86_64*) echo osx-64 ;;
    *linux*arm64*) echo linux-aarch64 ;;
    *linux*x86_64*) echo linux-64 ;;
  esac
}

BIN_LOCATION=bin/micromamba
[ -e "$BIN_LOCATION" ] && echo "$BIN_LOCATION already exists" >&2 || {
  url=https://micro.mamba.pm/api/micromamba/$(platform_name)/latest
  curl -sL $url | bunzip2 | tar xOf - bin/micromamba > "$BIN_LOCATION"
  chmod +x "$BIN_LOCATION"
}

env_type=$(echo "${1}" | tr "[:upper:]" "[:lower:]")
echo $env_type

if [[ $string == *osx* ]]; then
  # Use mac specific environment file for remote running
  os_postfix = "-mac"
fi

case $env_type in
  cuda|rocm|remote)
    bin/micromamba create -f environments/${env_type}${os_postfix}.yml -r runtime -n koboldai-${env_type} -y
    bin/micromamba create -f environments/${env_type}${os_postfix}.yml -r runtime -n koboldai-${env_type} -y
    ;;
  *)
    echo "Please specify either cuda or rocm or remote"
    ;;
esac
