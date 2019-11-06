#!/bin/bash

CMD=${1:-"cd /mask-rcnn-tensorflow; git pull; cd /; pip install --ignore-installed -e /mask-rcnn-tensorflow/"}
NAME=${2:-"mpicont"}

docker exec $NAME bash -c "$CMD"
