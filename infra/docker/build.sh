# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

BRANCH_NAME=${1:-"master"}
IMAGE_NAME=${2:-"mask-rcnn-tensorflow:latest"}

# The BRANCH_NAME refers to the git pull that happens inside of the Dockerfile
echo "Building docker image ${IMAGE_NAME} on branch ${BRANCH_NAME}"
echo ""



docker build -t ${IMAGE_NAME} ../.. --build-arg CACHEBUST=$(date +%s) --build-arg BRANCH_NAME=${BRANCH_NAME}
