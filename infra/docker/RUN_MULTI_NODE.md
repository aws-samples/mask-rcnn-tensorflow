## To run on multi-node
Make sure you have your data ready as in [Run with docker](https://github.com/aws-samples/mask-rcnn-tensorflow/blob/master/infra/docker/docker.md#using-docker "Run with docker").
### Create a AMI that support multi-node container training, Ubuntu is used as an example
- If you want to use EFA, follow the [Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html) to install EFA. EFA is only supported for P3dn.24xl at this time. Our example will use EFA.
  - You need to have your `aws-ofi-nccl` folder at home directory
  - (Optional) clone and build [nccl test](https://github.com/NVIDIA/nccl-tests)
- Create user keys
  - Create ssh keys use `ssh-keygen` and save the key in `~/.ssh/id_rsa`
  - Create a config file for ssh entry using the key at `~/.ssh/config` with the following content
  ```
  Host *
      StrictHostKeyChecking no
      UserKnownHostsFile=/dev/null
      LogLevel=ERROR
      ServerAliveInterval=30
  User ubuntu
  ```
  - Append​ the authorized_keys file with the public key with
  ```
  echo `cat id_rsa.pub` >> authorized_keys
  ```
- Setting up container communication
  - Create a directory to mount as /root/.ssh when running your container.
  ```
  mkdir ssh_container
  cd ssh_container
  ssh-keygen -t rsa
  Enter file in which to save the key (/home/ubuntu/.ssh/id_rsa):
  <path/to>/ssh_container/id_rsa
  ```
  - Create a config file in ssh_container folder with same content the `~/.ssh/config`
  - Copy the contents of the container's key `ssh_container/id_rsa.pub` to `ssh_container/authorized_keys`
  - Check and change permission of files as needed in ssh_container directory to match the following
  ```
  drwxrwxr-x  2 ubuntu ubuntu 4096 Sep 13 17:42 ./
  drwxr-xr-x 19 ubuntu ubuntu 4096 Sep 25 01:32 ../
  -rw-rw-r--  1 ubuntu ubuntu  410 Sep 12 23:27 authorized_keys
  -rw-rw-r--  1 root   root    131 Sep 13 17:42 config
  -rw-------  1 ubuntu ubuntu 1679 Sep 12 23:27 id_rsa
  -rw-r--r--  1 ubuntu ubuntu  410 Sep 12 23:27 id_rsa.pub
  ```
  - Change ownership of the `ssh_container/config` file by `sudo chown root:root config`. In the container you are the root user.
  - Create a file in the `~/.ssh` to execute commands inside the container by `vi /home/ubuntu/.ssh/mpicont.sh`. Add the following contents
  ```
  #!/bin/bash
  echo “entering container”

  # mpicont is your container name. You need to change it if you use a different name
  docker exec mpicont /bin/bash -c "$SSH_ORIGINAL_COMMAND"
  ```
  - Add execution permissions to `mpicont.sh` by `chmod +x /home/ubuntu/.ssh/mpicont.sh`
  - Once there is a ssh connection using the key we created specially for container, the commands should be run in the container. Add the following command to the end of `/home/ubuntu/.ssh/authorized_keys`
  ```
  command="bash $HOME/.ssh/mpicont.sh",no-port-forwarding,no-agent-forwarding,no-X11-forwarding <paste in contents of ssh_container/id_rsa.pub>
  ```
- Save this instance as an AMI
  - (Optional but highly recommended) Install [ClusterShell](https://clustershell.readthedocs.io/en/latest/index.html). It allows you to run some commands on all nodes in your cluster easily.
  - Install other packages you need
- Launch the cluster using the AMI you created, SSH into your master node
- Run the following commands on all nodes (clush can be useful here)
  - pull image `docker pull $IMAGE_NAME`
  - clone the repo`git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git`
  - launch container `./mask-rcnn-tensorflow/infra/docker/run_docker_efa.sh $IMAGE_NAME`. The script will launch container and let it sleep by default.
    - If you use clush, wait for all container to get started and use ctrl-c to quit clush.
    - You can mount other volumes from host to container by using `-v`.
    - You need to have your hostfile prepared in the container.
    - In this example we have EFA libs and nccl-test mounted to the container, change the `/mask-rcnn-tensorflow/infra/docker/run_docker_efa.sh` according to your needs
- Get into the container in your ***master node only*** with `docker exec -it mpicont bash`. Again change the mpicont if you changed your container name
- Launch the training with `/mask-rcnn-tensorflow/infra/train_multinode.sh $NUM_GPUS $BS_PER_GPU $PATH_TO_HOSTFILE`
