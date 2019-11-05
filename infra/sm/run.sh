#!/usr/bin/env bash
cd /opt/ml/code/mask-rcnn-tensorflow
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
	DATA.BASEDIR=/opt/ml/input/data/train \
	DATA.TRAIN='["train2017"]' \
	DATA.VAL='("val2017",)' \
	TRAIN.BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU} \
	TRAIN.LR_EPOCH_SCHEDULE='[(9, 0.1), (12, 0.01), (13, None)]' \
	TRAIN.EVAL_PERIOD=12 \
	BACKBONE.WEIGHTS=/opt/ml/input/data/train/pretrained-models/ImageNet-R50-AlignPadding.npz \
  RPN.TOPK_PER_IMAGE=True \
  PREPROC.PREDEFINED_PADDING=True \
  TRAIN.GRADIENT_CLIP=0 \
	TRAIN.WARMUP_INIT_LR=0.000416666666667 \
	FRCNN.BBOX_REG_WEIGHTS='[20., 20., 10., 10.]' \
	RPN.SLOW_ACCURATE_MASK=False \
	BACKBONE.NORM=FreezeBN \
	TRAINER=horovod
