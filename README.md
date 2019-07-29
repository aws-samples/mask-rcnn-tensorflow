# Mask RCNN

Performance focused implementation of Mask RCNN based on the [Tensorpack implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).
The original paper: [Mask R-CNN](https://arxiv.org/abs/1703.06870)
### Overview

This implementation of Mask RCNN is focused on increasing training throughput without sacrificing any accuracy. We do this by training with a batch size > 1 per GPU using FP16 and two custom TF ops.

### Status

Training on N GPUs (V100s in our experiments) with a per-gpu batch size of M = NxM training

Training converges to target accuracy for configurations from 8x1 up to 32x4 training. Training throughput is substantially improved from original Tensorpack code.

A pre-built dockerfile is available in DockerHub under `armandmcqueen/tensorpack-mask-rcnn:master-latest`. It is automatically built on each commit to master.

### Notes

- Running this codebase requires a custom TF binary - available under GitHub releases (custom ops and fix for bug introduced in TF 1.13
- We give some details the codebase and optimizations in `CODEBASE.md`

### To launch training
- Data preprocessing
  - Follow the [data preprocess](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)
  - If you want to use EKS or Sagemaker, you need to create your own S3 bucket which contains the data, and change the S3 bucket name in the following files:
    - EKS: [P3 config](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/eks/fsx/p3/stage-data.yaml), [P3dn config](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/eks/fsx/p3dn/stage-data.yaml)
    - SageMaker: [S3 download](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/sm/run_mpi.py#L122)
- Container is recommended for training
  - To train with docker, refer to [Docker](https://github.com/armandmcqueen/tensorpack-mask-rcnn/tree/master/infra/docker)
  - To train with Amazon EKS, refer to [EKS](https://github.com/armandmcqueen/tensorpack-mask-rcnn/tree/master/infra/eks)
  - To train with Amazon SageMaker, refer to [SageMaker](https://github.com/armandmcqueen/tensorpack-mask-rcnn/tree/master/infra/sm)

### Training results
The result was running on P3dn.24xl instances using EKS.
12 epochs training:

| Num_GPUs x Images_Per_GPU | Training time | Box mAP | Mask mAP |
| ------------- | ------------- | ------------- | ------------- |
| 8x4 | 5.09h | 37.47% | 34.45% |
| 16x4 | 3.11h | 37.41% | 34.47% |
| 32x4 | 1.94h | 37.20% | 34.25% |

24 epochs training:

| Num_GPUs x Images_Per_GPU | Training time | Box mAP | Mask mAP |
| ------------- | ------------- | ------------- | ------------- |
| 8x4 | 9.78h | 38.25% | 35.08% |
| 16x4 | 5.60h | 38.44% | 35.18% |
| 32x4 | 3.33h | 38.33% | 35.12% |

### Tensorpack fork point

Forked from the excellent Tensorpack repo at commit a9dce5b220dca34b15122a9329ba9ff055e8edc6
