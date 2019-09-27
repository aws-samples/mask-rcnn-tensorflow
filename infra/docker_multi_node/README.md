## To run on multi-node
Make sure you have your data ready as in [Run with docker](https://github.com/aws-samples/mask-rcnn-tensorflow/blob/master/infra/docker/docker.md#using-docker "Run with docker").
### SSH settings
- ssh into your master node and clone the repo by `git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git`
- run `cd ~/mask-rcnn-tensorflow/infra/docker_multi_node/`
- create your hosts file without slots
- run `./preprocess.sh $YOUR_MASTER_IP $YOUR_HOST_FILE`, this will enable the passwordless ssh connection and build the container on each of the nodes
### Pull or build the container
- To pull the image on every host, run `./pull_image.sh $HOSTFILE $IMAGE_NAME`
  - Your docker image should have OpenSSH server installed
- To build the image from scratch, run `./build_multi.sh $HOSTFILE $IMAGE_NAME $BRANCH`
### Launch training
- Create your hosts file with slots, which contains all ips of your nodes (include the primary host). The format should be like:
```
127.0.0.1 slots=8
127.0.0.2 slots=8
127.0.0.3 slots=8
127.0.0.4 slots=8
```
This is 4 nodes, 8 GPUs per node.
- Put your host file in `~/data` folder, the host file will be copied into container
- Launch training with running `./run_multinode.sh $YOUR_MASTER_IP $YOUR_HOST_FILE $NUM_GPU $BS $JOB_NAME`
### Stop training
- If the training stop naturally, there is nothing need to do. The log will be in the `~/logs` folder
- If you want to stop the training early, you should stop the container in a separate shell or in one other machine.
