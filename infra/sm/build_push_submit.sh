# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
imagename=$1
sagemaker_iam_role=$2
./build_and_push.sh $imagename
python3 launch_sm_job.py $imagename $sagemaker_iam_role
