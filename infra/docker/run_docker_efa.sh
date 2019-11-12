#!/bin/bash
IMAGE=${1:-"fewu/mask-rcnn-tensorflow:master-latest"}
CMD=${2:-"sleep infinity"}
docker run --rm --gpus 8 --name mpicont \
        --net=host --uts=host --ipc=host \
        --ulimit stack=67108864 --ulimit memlock=-1 \
        --security-opt seccomp=unconfined \
        -v /opt/amazon/efa:/efa \
        -v /home/ubuntu/data:/data \
        -v /home/ubuntu/logs:/logs \
        -v /home/ubuntu/ssh_container/:/root/.ssh/ \
        -v /home/ubuntu/utils:/utils \
        -v /home/ubuntu/aws-ofi-nccl:/ofi \
        -v /home/ubuntu/nccl-cont/:/nccl-tests \
        --device=/dev/infiniband/uverbs0 \
        $IMAGE $CMD
