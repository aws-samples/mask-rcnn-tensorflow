MASTER_HOST=${1:-"127.0.0.1"}
BRANCH_NAME=${2:-"master"}
NUM_GPU=${3:-8}
BS=${4:-4}
HOSTS=${5:-"hosts"}
HOSTS_SLOTS=${6:-"hosts_slots"}


hosts=`cat $HOSTS`

for host in $hosts; do
  if [ $host != $MASTER_HOST ]; then
    ssh $host "screen -L -d -m bash -c \" nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs mask-rcnn-tensorflow:${BRANCH_NAME} /bin/bash -c /usr/sbin/sshd -p 1234; sleep infinity\""
  fi
done
nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs mask-rcnn-tensorflow:dev-multi-node-docker /bin/bash -c "cp /data/${HOSTS_SLOTS} /mask-rcnn-tensorflow/infra/docker_multi_node/hosts; cd /mask-rcnn-tensorflow/infra/docker_multi_node/; ./train_multinode.sh $NUM_GPU $BS |& tee /logs/${NUM_GPU}x${BS}.log"
for host in $hosts; do
  if [ $host != $MASTER_HOST ]; then
    echo "stop containers and screens..."
    docker stop $(docker ps -a -q)
    pkill screen
  fi
done
