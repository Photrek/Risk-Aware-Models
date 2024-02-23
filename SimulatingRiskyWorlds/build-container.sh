#!/usr/bin/bash

. ./config-vars.sh

starttime=$EPOCHSECONDS
docker build --build-arg USERNAME=$USER -t $imgName .
durmin=$((($EPOCHSECONDS-$starttime)/60))
dursec=$((($EPOCHSECONDS-$starttime)%60))
echo "Build duration: " $durmin ":" $dursec
