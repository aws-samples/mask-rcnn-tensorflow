# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

IMAGE_NAME=${1:-"mask-rcnn-tensorflow:latest"}

echo "Running docker image ${IMAGE_NAME}"
echo ""



nvidia-docker run -it  -v ~/data:/data -v ~/logs:/logs ${IMAGE_NAME}
