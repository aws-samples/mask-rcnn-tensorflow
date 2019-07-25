# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash

# Set timestamp and logging directory, begin writing to it.
TS=`date +'%Y%m%d_%H%M%S'`
LOG_DIR=/home/ubuntu/logs/train_log_${TS}
mkdir -p ${LOG_DIR}
exec &> >(tee ${LOG_DIR}/nohup.out)

# Print evaluated script commands
set -x

# Set VENV
VENV=${CONDA_DEFAULT_ENV}

# Write current branch and commit hash to log directory
git branch | grep \* | awk '{print $2}' > ${LOG_DIR}/git_info
git log | head -1 >> ${LOG_DIR}/git_info
git diff >> ${LOG_DIR}/git_info

# Copy this script into logging directory
cp `basename $0` ${LOG_DIR}

# Record environment variables
env > ${LOG_DIR}/env.txt

# Record python libaries
pip freeze > ${LOG_DIR}/requirements.txt

# Record tensorflow shared object linkages (CUDA version?)
ldd /home/ubuntu/anaconda3/envs/${VENV}/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so > ${LOG_DIR}/tf_so_links.txt

# Execute training job
# HOROVOD_TIMELINE=${LOG_DIR}/htimeline.json \
#HOROVOD_AUTOTUNE=1 \
#HOROVOD_AUTOTUNE_LOG=${LOG_DIR}/hvd_autotune.log \
HOROVOD_CYCLE_TIME=0.5 \
HOROVOD_FUSION_THRESHOLD=67108864 \
HOROVOD_LOG_LEVEL=INFO \
TENSORPACK_FP16=1 \
/home/ubuntu/anaconda3/envs/${VENV}/bin/mpirun -np 8 -H localhost:8 \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO \
-x HOROVOD_CYCLE_TIME \
-x HOROVOD_FUSION_THRESHOLD \
-x TENSORPACK_FP16 \
-x LD_LIBRARY_PATH -x PATH \
--output-filename ${LOG_DIR}/mpirun_logs \
/home/ubuntu/anaconda3/envs/${VENV}/bin/python3 /home/ubuntu/tensorpack-mask-rcnn/MaskRCNN/train.py \
--logdir ${LOG_DIR} \
--fp16 \
--throughput_log_freq 2000 \
--config MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/home/ubuntu/data \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.BATCH_SIZE_PER_GPU=1 \
TRAIN.LR_EPOCH_SCHEDULE='[(8, 0.1), (10, 0.01), (12, None)]' \
BACKBONE.WEIGHTS=/home/ubuntu/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAIN.EVAL_PERIOD=12 \
TRAINER=horovod

#For 32x4
#TRAIN.GRADIENT_CLIP=1.5

#-x HOROVOD_AUTOTUNE \
#-x HOROVOD_AUTOTUNE_LOG \
#-x HOROVOD_LOG_LEVEL=INFO \
#-x HOROVOD_CYCLE_TIME -x HOROVOD_FUSION_THRESHOLD \
#TRAIN.EVAL_PERIOD=1 \
#TRAIN.STEPS_PER_EPOCH=15000 \
#TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
