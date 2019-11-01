# Using Docker
The ec2 instance must have the training data available at ~/data. You also need to have docker and nvidia-docker installed on the AMI.

### Build container
- To use our docker image, run `docker pull fewu/mask-rcnn-tensorflow:master-latest`
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

## Notes

The current Dockerfile uses the wheel built for p3.16xl. The wheel built for p3dn.24xl might have a performance improvement, but it does not run on 16xl due to different available instruction sets.
