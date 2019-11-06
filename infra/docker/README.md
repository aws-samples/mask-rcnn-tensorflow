# Using Docker
The ec2 instance must have the training data available at ~/data. You also need to have Nvidia driver, docker and nvidia-docker installed on the AMI.

## To run with single node
### Build container
- To use our docker image, run `docker pull awssamples/mask-rcnn-tensorflow:latest`
- To build your own image (it will be slow), run
```
git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git
cd mask-rcnn-tensorflow/infra/docker
./build.sh
```

### Run container interactively
```
./run.sh
```

### Run training job inside container

```
cd mask-rcnn-tensorflow
infra/docker/train.sh 8 1 250
```

This is 8 GPUs, 1 img per GPU, summary writer logs every 250 steps.

Logs will be exposed to the ec2 instance at ~/logs.

### Attaching/Detaching from docker container
`ctl + p + q` will detach
`docker ps` will give info on the running docker containers including convenient name.
`docker attach $CONTAINER_NAME` will reattach to the running docker container.
`docker exec -it $CONTAINER_NAME bash` will start a separate terminal.

## To run with multi-node
- refer to [RUN_MULTI_NODE.md](https://github.com/aws-samples/mask-rcnn-tensorflow/blob/master/infra/docker/RUN_MULTI_NODE.md)
