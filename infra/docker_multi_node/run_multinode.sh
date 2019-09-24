MASTER_HOST=${1:-"127.0.0.1"}
BRANCH_NAME=${2:-"master"}
HOSTS=${3:-"hosts"}
HOSTS_SLOTS=${4:-"hosts_slots"}


hosts=`cat $HOSTS`

for host in $hosts; do
  if [ $host != $MASTER_HOST ]; then
    ssh $host "nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs mask-rcnn-tensorflow:dev-${BRANCH_NAME} /mask-rcnn-tensorflow/infra/docker_multi_node/sleep.sh"
  fi
done
nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs mask-rcnn-tensorflow:dev-${BRANCH_NAME} "cp /data/${HOSTS_SLOTS} /mask-rcnn-tensorflow/infra/docker_multi_node/hosts; cd /mask-rcnn-tensorflow/infra/docker_multi_node/; ./train_multinode.sh $NUM_GPU $BS |& tee /logs/${NUM_GPU}x${BS}.log"

for host in $hosts; do
  docker stop $(docker ps -a -q)
done
