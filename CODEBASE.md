# Codebase Details

For training, the codebase is broken into a few major parts:

- `train.py` handles parsing the command line arguments, loading the model, and configuring and launching the training.
- `config.py` holds various configurations needed by the model.
- `dataset.py` handles a lot of COCO specific logic such as parsing annotations and computing evaluation metrics from a set of predictions.
- `common.py`, `model_box.py`, `performance.py`, `viz.py` and `utils/` hold various utilities, mostly for either debugging or manipulating bounding boxes, anchors and segmentation masks.

The core model has two components - the Tensorflow graph and the non-Tensorflow data pipelines implemented using Tensorpack DataFlows and numpy.

- Code for the Tensorflow graph is in `model/`.
    - `model/generalized_rcnn.py` has a `DetectionModel` class whose `build_graph()` method outlines a generic two stage detector model.
    - `ResNetFPNModel` subclasses that and provides implementations of the backbone, rpn and roi_heads. `ResNetFPNModel.inputs()` uses tf.placeholders to define the input it expects from the dataflow
- `data.py` holds the data pipeline logic. The key function for training is `get_batch_train_dataflow()`, which contains logic for reading data from disk and batching datapoints together.

# General Optimizations

## Batchification

We added the ability have a per-GPU batch size greater than 1. Below are some implementation details of what that means.

- In the data pipeline, we took the existing outputs (image, gt_label, gt_boxes, gt_masks) and added a batch dimension, padding when different images have different shapes. To account for this padding in the model, we added two new inputs, `orig_image_dims` and `orig_gt_counts` which are used to slice away padding when necessary.
- Most of the core convolutional and fully connected layers used tf.Layers which already supports a batch dimension, so we didn't need to make substantial changes there. We did add FP16 support for most layers that use matrix multiplications, so take advantage of NVIDIA's tensor cores.
- In some cases, such as calculating losses, batchification meant looping over each image in the per-GPU batch, calculating the loss for a single image and then averaging the losses over the per-GPU batch.
    - This may be an area to improve performance in the future, but we are unsure how impactful optimizing that code would be as the operations where we do this seem to be fairly computationally lightweight.

## Custom TF ops

We have used two custom Tensorflow ops to get improved throughput. This is the main reason you need to use a custom Tensorflow binary (this is also a bug introduced in TF 1.13 that we needed to fix, see `patch/`).

In the RPN, we use `tf.generate_bounding_box_proposals` which is a function that takes in the RPN logits, converts the regression logits into bounding box coordinates and then applies NMS.

In the ROI heads, we use `tf.roi_align`.

Both of these are implemented here: https://github.com/samikama/tensorflow/commits/GenerateProposalsOp

These ops is being upstreamed to TF (with some minor differences) and we will move this codebase to the native TF ops once we are able.

- ROIAlign PR - https://github.com/tensorflow/tensorflow/pull/28746
- Generate Bounding Box PR - https://github.com/tensorflow/tensorflow/pull/28754

## FP16

We offer mixed precision training, which substantially improves throughput on NVIDIA tensor cores. It is important to note that you need to both pass in the `--fp16` flag to `train.py` AND set the `TENSORPACK_FP16` environmental variable to 1. `--fp16` tells the model to use FP16, while the `TENSORPACK_FP16` envvar enables loss scaling.

Loss scaling occurs inside of the Tensorpack library, not in the MaskRCNN model code, which is why you need to set both. This should probably be addressed in the future.

## NHWC convolution kernels

According to Nvidia's update in MLPerf v0.6 - https://devblogs.nvidia.com/nvidia-boosts-ai-performance-mlperf-0-6/, tensor core accelerated convolution kernels expect NHWC (or “channels-last) layout. We added options to transpose convolution input data format to NHWC for backbone, FPN, RPN-head and mask-head.

For tensorflow, NHWC kernel can only be activated by XLA or in Nvidia's NGC container. Since XLA doesn't work in our case (our tensor shapes are dynamic and tensorpack tower function can not be compiled with XLA), we use NGC container and export `TF_ENABLE_NHWC=1`

## Disable Cudnn autotune and use aspect ratio grouping

We disabled the cudnn autotune by export `TF_CUDNN_USE_AUTOTUNE=0` since the cudnn warmup is pretty slow and tuning result does not deliver a significant improve.

When training with bs > 1, if the images sizes differ a lot in the same batch, padding will introduce extra computations which are not negligible. The idea is to batch the images of the same aspect ratio, i.e. height > width or height < width.

Reference: https://github.com/facebookresearch/Detectron/blob/35fede2bd11176c600617e60ea4e7cc458cf4d64/detectron/roi_data/loader.py#L139-L157

## Async evaluation and faster coco evaluation

MLPerf v0.6 requires to evaluate every epoch and stop the training once the target accuracy is reached (box/mask 0.377/0.339). We start a background thread overlap the coco eval with the next iteration computation. Use the `--async_eval` to enable this feature.

The coco eval usually takes more than 2 minutes. This will be a bottleneck for 24 nodes when each epoch takes less than 2 minutes. We adopt the [Nvidia's own implementation](https://github.com/NVIDIA/cocoapi/) of coco eval, which cut the time to less than 10 seconds.

## Determinism

Determinism can help to converge to a more stable result across different runs. It is also helpful for debugging purposes. By setting up op level seed for the TF graph and a generator for other random numbers, we have fully determinism for the forward computation. However for backprop, the ROIAlign op's CUDA kernel uses `atomicAdd` which introduces randomness. In order to maintain determinism, the TF_CUDNN_USE_AUTOTUNE should be set to 0. This is the default in our Docker container.

## Miscellaneous
 - Add gradient clipping to avoid loss going NaN when global batch >= 128
 - Faster coco loading from https://github.com/tensorpack/tensorpack/commit/8c8de86c46cadebb1860feae832347e423f5942b
 - Improved accuracy via better anchor generation and more accurate mask paste from https://github.com/tensorpack/tensorpack/commit/141ab53cc37dce728802803747584fc0fb82863b
 - To enable our focus on throughput, we have removed some unused features from the original Tensorpack code
    - We only support FPN-based training
    - We do not have support for Cascade RCNN

# AWS specific optimizations

## Using EFA for faster communication

Elastic Fabric Adapter (EFA) is a network interface for Amazon EC2 instances that enables customers to run applications requiring high levels of inter-node communications at scale on AWS. EFA works together with MPI (Message Passing Interface) and NCCL (NVIDIA Collective Communications Library) make all cross node communication faster.

Reference: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

## Using CPU core binding to increase the CPU utilization

There are 64 CPU cores for P3.16xl and 96 CPU cores for P3dn.24xl. When we train with Horovod we have 1 process for each of the GPU. However, some processes may race for the same CPU cores if we don't schedule well.

Our binding script ([P3](https://github.com/aws-samples/mask-rcnn-tensorflow/blob/master/infra/docker/ompi_bind_p3_16.sh), [P3dn](https://github.com/aws-samples/mask-rcnn-tensorflow/blob/master/infra/docker/ompi_bind_p3dn.sh)) will evenly distribute the cores among all processes which increase the CPU utilization

# TODO

## Tensorpack changes since fork we may want to port
- Port TensorSpec changes, replacing tf.placeholder
- Change to L1 loss? - https://github.com/tensorpack/tensorpack/commit/d263818b8fe8d8e096c4826dc5f2432901c5a894#diff-75814de28422d125213d581d1a36d92a

## Fully determinism

## Support for TF2.0
