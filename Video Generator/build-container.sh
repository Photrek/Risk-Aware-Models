#!/usr/bin/bash
starttime=$EPOCHSECONDS
docker build --build-arg USERNAME=$USER -t ptl-base .
durmin=$((($EPOCHSECONDS-$starttime)/60))
dursec=$((($EPOCHSECONDS-$starttime)%60))
echo "Build duration: " $durmin ":" $dursec
