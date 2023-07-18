# Codebase Details

For training, the codebase is broken into a few major parts:

* `train.py` handles parsing the command line arguments, loading the model, and configuring and launching the training.
* `config.py` holds various configurations needed by the model.
* `dataset.py` handles a lot of COCO specific logic such as parsing annotations and computing evaluation metrics from a set of predictions.
* `common.py`, `model_box.py`, `performance.py`, `viz.py` and `utils/` hold various utilities, mostly for either debugging or manipulating bounding boxes, anchors and segmentation masks.

The core model has two components - the Tensorflow graph and the non-Tensorflow data pipelines implemented using Tensorpack DataFlows and numpy.

* Code for the Tensorflow graph is in `model/`.
    - `model/generalized_rcnn.py` has a `DetectionModel` class whose `build_graph()` method outlines a generic two stage detector model.
    - `ResNetFPNModel` subclasses that and provides implementations of the backbone, rpn and roi_heads. `ResNetFPNModel.inputs()` uses `tfv1.placeholders` to define the input it expects from the dataflow
* `data.py` holds the data pipeline logic. The key function for training is `get_batch_train_dataflow()`, which contains logic for reading data from disk and batching datapoints together.

## General Optimizations

### Adding batch dimension 

We added the ability have a per-GPU batch size greater than 1. Below are some implementation details of what that means:

* In the data pipeline, we took the existing outputs (image, gt_label, gt_boxes, gt_masks) and added a batch dimension, padding when different images have different shapes. To account for this padding in the model, we added two new inputs, `orig_image_dims` and `orig_gt_counts` which are used to slice away padding when necessary.
* The model only supports the [Feature Pyramid Network (FPN)](https://arxiv.org/pdf/1612.03144.pdf) option for the ResNet backbone. We do  not support Cascade RCNN.
* Region Proposal Network (RPN) proposals for each FPN level are generated with batch dimension.
* ROI Align features are computed with batch dimension.

## AWS specific optimizations

### Using EFA for faster network communication

Elastic Fabric Adapter (EFA) is a network interface for Amazon EC2 instances that enables customers to run applications requiring high levels of inter-node communications at scale on AWS. EFA works together with MPI (Message Passing Interface) and NCCL (NVIDIA Collective Communications Library) make all cross node communication faster.

Reference: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

