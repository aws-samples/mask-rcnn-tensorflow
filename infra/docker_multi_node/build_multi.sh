# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

HOSTS=${1:-"hosts"}
BRANCH_NAME=${2:-"master"}

hosts=`cat $HOSTS`

for host in $hosts; do
    ssh $host "cd ~/mask-rcnn-tensorflow; git checkout $BRANCH_NAME; git pull"
    ssh $host 'bash --login -c "screen -L -d -m bash -c \" cd /home/ubuntu/mask-rcnn-tensorflow/infra/docker; ./build.sh\""'
done
