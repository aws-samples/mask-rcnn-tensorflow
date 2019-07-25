# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
cd /opt/ml/code/tensorpack-mask-rcnn
BATCH_SIZE_PER_GPU=4
THROUGHPUT_LOG_FREQ=2000
echo "Launch training job...."
/usr/local/bin/python3 MaskRCNN/train.py \
	--logdir /logs/train_log \
	--fp16 \
	--throughput_log_freq ${THROUGHPUT_LOG_FREQ} \
	--config \
	MODE_MASK=True \
	MODE_FPN=True \
	DATA.BASEDIR=/opt/ml/code/data \
	DATA.TRAIN='["train2017"]' \
	DATA.VAL='("val2017",)' \
	TRAIN.BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} \
	TRAIN.LR_EPOCH_SCHEDULE='[(8, 0.1), (10, 0.01), (12, None)]' \
	TRAIN.EVAL_PERIOD=12 \
	BACKBONE.WEIGHTS=/opt/ml/code/data/pretrained-models/ImageNet-R50-AlignPadding.npz \
  RPN.TOPK_PER_IMAGE=True \
  PREPROC.PREDEFINED_PADDING=True \
  TRAIN.GRADIENT_CLIP=0 \
	BACKBONE.NORM=FreezeBN \
	TRAINER=horovod
