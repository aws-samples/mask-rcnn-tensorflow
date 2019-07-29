# To train with docker

## To run on single-node
Refer to [Run with docker](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/docker/docker.md#using-docker "Run with docker")

## To run on multi-node
Make sure you have your data ready as in [Run with docker](https://github.com/armandmcqueen/tensorpack-mask-rcnn/blob/master/infra/docker/docker.md#using-docker "Run with docker").
### SSH settings and build container
- ssh into your master node and clone the repo by `git clone https://github.com/armandmcqueen/tensorpack-mask-rcnn.git`
- run `cd ~/tensorpack-mask-rcnn/infra/docker/`
- create your hosts file without slots
- run `./ssh_and_build.sh $YOUR_MASTER_IP $YOUR_HOST_FILE`, this will enable the passwordless ssh connection and build the container on each of the nodes
### run container
For each of the instances
- run `cd ~/tensorpack-mask-rcnn/infra/docker/`
- run the container by run `./run_multinode.sh`

### Launch training
Inside the container:
- On each host *apart from the primary* run the following in the container you started:
  - run `cd tensorpack-mask-rcnn/infra/docker/`
  - run `./sleep.sh`
This will make those containers listen to the ssh connection from port 1234.
- On primary host, `cd tensorpack-mask-rcnn/infra/docker`, create your hosts file, which contains all ips of your nodes (include the primary host). The format should be like:
```
127.0.0.1 slots=8
127.0.0.2 slots=8
127.0.0.3 slots=8
127.0.0.4 slots=8
```
This is 4 nodes, 8 GPUs per node.
Launch training with running `./train_multinode.sh 32 4` for 32 GPUs and 4 images per GPU
