#!/usr/bin/bash
docker run \
  --interactive \
  --tty \
  --rm \
  --gpus=all \
  --mount type=bind,source="$(pwd)",target=/home \
  --workdir=/home \
  --name cntr-ptl-vaernn-$RANDOM \
  ptl-base \
  bash --init-file <(echo ". \"$HOME/.bashrc\"; set -o vi")

#  bash --init-file <(echo "set -o vi")
#  'bash --init-file <("set -o vi")'
#  -p 8888:8888 \
#  nvidia/cuda:12.0.1-base-ubuntu22.04 \
#  -p 18088:8088 \
#8088 in container to 18088 in host
 
#  --name "$USER"-cntr-pyto \
