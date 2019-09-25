# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

MASTER_HOST=${1:-"127.0.0.1"}
HOSTS=${2:-"hosts"}
HOSTS_SLOTS=${2:-"hosts_slots"}
BRANCH_NAME=${3:-"master"}


ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa
hosts=`cat $HOSTS`
for host in $hosts; do
  scp ~/.ssh/id_rsa.pub $host:~/.ssh/
  ssh $host "cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys"
  ssh $host "printf 'Host *\n  StrictHostKeyChecking no\n' >> ~/.ssh/config"
  ssh $host "chmod 400 ~/.ssh/config"
  ssh $host "sudo mkdir -p /mnt/share/ssh"
  ssh $host "sudo cp -r ~/.ssh/* /mnt/share/ssh"
  if [ $host != $MASTER_HOST ]; then
    ssh $host "git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git -b ${BRANCH_NAME}"
  fi
  ssh $host "cd ~/mask-rcnn-tensorflow/infra/docker; ./build.sh ${BRANCH_NAME}"
  if [ $host != $MASTER_HOST ]; then
    ssh $host "nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs mask-rcnn-tensorflow:dev-${BRANCH_NAME} /mask-rcnn-tensorflow/infra/docker/sleep.sh"
  fi
done
nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs mask-rcnn-tensorflow:dev-${BRANCH_NAME} "cp /data/${HOSTS_SLOTS} /mask-rcnn-tensorflow/infra/docker/hosts; cd /mask-rcnn-tensorflow/infra/docker/; ./train_multinode.sh $NUM_GPU $BS |& tee /logs/${NUM_GPU}x${BS}.log"
