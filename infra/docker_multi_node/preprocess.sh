MASTER_HOST=${1:-"127.0.0.1"}
HOSTS=${2:-"hosts"}
BRANCH_NAME=${3:-"master"}

ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa
hosts=`cat $HOSTS`
for host in $hosts; do
  scp ~/.ssh/id_rsa.pub $host:~/.ssh/
  ssh $host "cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys"
  ssh $host "printf 'Host *\n  StrictHostKeyChecking no\n' >> ~/.ssh/config"
  ssh $host "chmod 400 ~/.ssh/config"
  ssh $host "sudo mkdir -p /mnt/share/ssh"
  ssh $host "sudo cp -r ~/.ssh/* /mnt/share/ssh"
  if [ $host != $MASTER_HOST ]; then
    ssh $host "git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git -b ${BRANCH_NAME}"
  fi
done
