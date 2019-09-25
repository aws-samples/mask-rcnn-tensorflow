# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

HOSTS=${1:-"hosts"}
IMAGE_NAME=${2:-"fewu/mask-rcnn-tensorflow:master-latest"}
BRANCH_NAME=${3:-"master"}

hosts=`cat $HOSTS`

for host in $hosts; do
    ssh $host "cd ~/mask-rcnn-tensorflow; git checkout $BRANCH_NAME; git pull"
    ssh $host 'bash --login -c "screen -L -d -m bash -c \" cd /home/ubuntu/mask-rcnn-tensorflow;'\
     ' docker build -t '"${IMAGE_NAME}"' . --build-arg CACHEBUST=$(date +%s) --build-arg BRANCH_NAME='"${BRANCH_NAME}"'\""'
done
