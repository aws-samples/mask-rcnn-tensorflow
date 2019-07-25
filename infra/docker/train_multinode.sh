# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
NUM_GPU=${1:-1}
BATCH_SIZE_PER_GPU=${2:-1}
PORT_ID=${3:-1234}
THROUGHPUT_LOG_FREQ=${4:-2000}



echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "BATCH_SIZE_PER_GPU: ${BATCH_SIZE_PER_GPU}"
echo "THROUGHPUT_LOG_FREQ: ${THROUGHPUT_LOG_FREQ}"
echo ""



/usr/local/bin/mpirun -np ${NUM_GPU} \
--hostfile hosts \
--mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_exclude lo,docker0 \
-mca btl_vader_single_copy_mechanism none \
-mca plm_rsh_args "-p ${PORT_ID}" \
-x LD_LIBRARY_PATH \
-x PATH \
-x NCCL_SOCKET_IFNAME=^docker0,lo \
-x NCCL_MIN_NRINGS=8 \
-x NCCL_DEBUG=INFO \
-x TENSORPACK_FP16=1 \
-x HOROVOD_CYCLE_TIME=0.5 \
-x HOROVOD_FUSION_THRESHOLD=67108864 \
--output-filename /logs/mpirun_logs \
/usr/local/bin/python3 /tensorpack-mask-rcnn/MaskRCNN/train.py \
--logdir /logs/train_log \
--fp16 \
--throughput_log_freq ${THROUGHPUT_LOG_FREQ} \
--config \
MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=/data \
DATA.TRAIN='["train2017"]' \
DATA.VAL='("val2017",)' \
TRAIN.BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} \
TRAIN.LR_EPOCH_SCHEDULE='[(8, 0.1), (10, 0.01), (12, None)]' \
TRAIN.EVAL_PERIOD=12 \
RPN.TOPK_PER_IMAGE=True \
PREPROC.PREDEFINED_PADDING=True \
BACKBONE.WEIGHTS=/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=FreezeBN \
TRAINER=horovod
#For 32x4
#TRAIN.GRADIENT_CLIP=1.5
