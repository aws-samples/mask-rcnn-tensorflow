MASTER_HOST=${1:-"127.0.0.1"}
IMAGE_NAME=${2:-"fewu/mask-rcnn-tensorflow:master-latest"}
NUM_GPU=${3:-8}
BS=${4:-4}
JOB_NAME=${5:-"8x4-run1"}
HOSTS=${5:-"hosts"}
HOSTS_SLOTS=${6:-"hosts"}


hosts=`cat $HOSTS`

for host in $hosts; do
  if [ $host != $MASTER_HOST ]; then
    ssh $host 'screen -L -d -m bash -c "nvidia-docker run'\
    ' --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs' "${IMAGE_NAME}"\
    ' /bin/bash -c \" cd /mask-rcnn-tensorflow; git pull; /usr/sbin/sshd -p 1234; sleep infinity \""'
  fi
done
nvidia-docker run --network=host -v /mnt/share/ssh:/root/.ssh -v ~/data:/data -v ~/logs:/logs ${IMAGE_NAME} /bin/bash -c "cp /data/${HOSTS_SLOTS} /mask-rcnn-tensorflow/infra/docker_multi_node/hosts; cd /mask-rcnn-tensorflow/infra/docker_multi_node/; git pull; ./train_multinode.sh $NUM_GPU $BS |& tee /logs/${JOB_NAME}.log"

for host in $hosts; do
  if [ $host != $MASTER_HOST ]; then
    echo "stop containers and screens on ${host}..."
    ssh $host 'screen -L -d -m bash -c "docker ps -q | xargs -r docker stop; pkill screen"'
  fi
done
