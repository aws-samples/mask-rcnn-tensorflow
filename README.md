# Mask R-CNN

This is an optimized version of [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) based on TensorFlow 2.x, and [Tensorpack Faster R-CNN/Mask R-CNN on COCO](https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/README.md) implementation.

### Overview

This implementation of Mask R-CNN is focused on increasing training throughput without sacrificing accuracy. We do this by training with a per-GPU batch size > 1. [Tensorpack Faster R-CNN/Mask R-CNN on COCO](https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/README.md) implementation only supports a per-GPU batch size of 1. 

This implementation **does not** make use of any custom TensorFlow Ops.

This implementation supports [Horovod](https://github.com/horovod/horovod) for multi-node, multi-GPU distributed training. 

### Training convergence

Training on N GPUs with a per-gpu batch size of M = NxM total batch size.

Training converges to target accuracy for configurations from 8x1 up to 32x4 training. Training throughput is substantially improved from original Tensorpack code. 

### Training data

* We are using COCO 2017, you can download the data from [COCO data](http://cocodataset.org/#download).
* The pre-trained resnet backbone can be downloaded from [ImageNet-R50-AlignPadding.npz](http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz)
* The file folder needs to have the following directory structure:

```
  data/
    annotations/
      instances_train2017.json
      instances_val2017.json
    pretrained-models/
      ImageNet-R50-AlignPadding.npz
    train2017/
      # image files that are mentioned in the corresponding json
    val2017/
      # image files that are mentioned in corresponding json
```

### Login to Amazon Elastic Container Registry (ECR)

[Dockerfile](./Dockerfile) used in this project is based on AWS [Deep-learning Container Images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md). Before you can build the Dockerfile, You must login into [Amazon Elastic Container Registry (ECR) ](https://aws.amazon.com/ecr/) in `us-west-2` using following command in your Visual Studio Code Terminal:

    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

When your AWS session expires, you will need to login again.

### Launch training in Visual Studio Code

For training on a GPU enabled desktop, we recommend using [Visual Studio Code](https://code.visualstudio.com/). Install Python and Docker extensions for the Visual Studio Code.

The `docker-run: debug` task defined in [tasks.json](./.vscode/tasks.json) runs training in a Docker container using the docker extension. The Docker image for the task is built automatically using [Dockerfile](./Dockerfile) when you run the task. This Visual Studio Code task enables debugging, as well. 

Configure Docker container volumes `localPath` in [tasks.json](./.vscode/tasks.json). Configure training script configuration in [launch.json](./.vscode/launch.json). 

### Launch notebooks in Visual Studio Code

We provide two Jupyer notebooks:

* For visualization of trained model: [visualization.ipynb](./notebooks/visualization.ipynb)
* For inspection of COCO data: [coco_inspection.ipynb](./notebooks/coco_inspection.ipynb)

The `docker-run: notebooks` Visual Studio Code task defined in [tasks.json](./.vscode/tasks.json) runs Jupyter Lab in a Docker container using the docker extension. The Docker image for the task is built automatically using [Dockerfile](./Dockerfile) when you run the task. 

Configure Docker container volumes `localPath` in [tasks.json](./.vscode/tasks.json). 

When you run the `docker-run: notebooks` Visual Studio Code task, the Jupyter Lab server runs in a detached container. To connect to the Jupyter Lab notebook in your browser, execute following steps:

* When you run the `docker-run: notebooks` task in Visual Studio Code, you will see the `container-id`   printed in the Visual Studio Code Terminal.
* Use the `container-id` to connect to the container using following command in a terminal: `docker exec -it container-id /bin/bash`
* At the shell prompt inside the container, run the following command: `cat nohup.out`

This will print the instructions for connecting to the Jupyter Lab server in a browser. When you are done using the notebooks, close the browser window, and stop the Jupyter Lab container using the command: `docker stop container-id`.

### Tensorpack compatibility

This implementation was originally forked from the [Tensorpack](https://github.com/tensorpack/tensorpack) repo at commit `a9dce5b220dca34b15122a9329ba9ff055e8edc6`. Tensorpack code in this repo has been updated since the original fork to support TensorFlow 2.x, and is approximately equivalent to Tensorpack commit `fac024f0f72fd593ea243f0b599a51b11fe4effd`. 

## Codebase

See [Codebase](./CODEBASE.md) for details about the code.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the [LICENSE](./LICENSE).
