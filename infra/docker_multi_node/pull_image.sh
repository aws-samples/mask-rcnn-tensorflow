#!/usr/bin/env bash

HOSTS=${1:-"hosts"}
IMAGE_NAME=${2:-"fewu/mask-rcnn-tensorflow:master-latest"}

hosts=`cat $HOSTS`

for host in $hosts; do
    ssh $host 'bash --login -c "screen -L -d -m bash -c \" cd /home/ubuntu/mask-rcnn-tensorflow; docker pull '"${IMAGE_NAME}"' \""'
done
