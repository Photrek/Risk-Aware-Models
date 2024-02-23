#!/usr/bin/bash

# Load config-vars
. ./config-vars.sh
# Note: the above laods random_container.sh

# Recreate the random_container.sh file with new value for random_container.
# This is done because we're about to conusme random_container value w/in "docker run ...".
echo random_container=$RANDOM > random_container.sh

echo The container about to be launched: $containerName-$random_container

docker run \
  --interactive \
  --tty \
  --rm \
  --gpus=all \
  --mount type=bind,source="$(pwd)",target=/home \
  --workdir=/home \
  --name $containerName-$random_container \
  $imgName \
  bash

#  bash --init-file <(echo ". \"$HOME/.bashrc\"; set -o vi")
#  bash --init-file <(echo "set -o vi")
#  'bash --init-file <("set -o vi")'
#  -p 8888:8888 \
#  nvidia/cuda:12.0.1-base-ubuntu22.04 \
#  -p 18088:8088 \
#8088 in container to 18088 in host
 
#  --name "$USER"-cntr-pyto \
