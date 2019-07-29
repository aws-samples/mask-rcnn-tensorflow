# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
import time
import subprocess
import json
import sagemaker_containers.beta.framework as framework

from contextlib import contextmanager
import signal
import paramiko


logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)


def _change_hostname(current_host):
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.
    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("change-hostname.sh {}".format(current_host))

def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])

def _wait_master_to_finish(master_host):
    while _can_connect(master_host):
        time.sleep(30)
    print("Lose connection with master, quit")

def _wait_master_to_start(master_host):
    while not _can_connect(master_host):
        time.sleep(1)
    print("Establish connection with master")

def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            logger.info("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                if _can_connect(host):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port=22):
    """
     Checks if the connection to provided ``host`` and ``port`` is possible or not.
     Args:
        host (str): Hostname for the host to check connection.
        port (int): Port name of the host to check connection on.
    """
    try:
        logger.debug('Testing connection to host %s', host)
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host,
                       port=port)
        client.close()
        logger.info('Can connect to host %s', host)
        return True
    except Exception as e:
        logger.info('Cannot connect to host %s', host)

        logger.info('Connection failed with exception: \n %s', str(e))
        return False


def train(hosts, current_host, num_gpus, custom_mpi_cmds):
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)
    process_slots_per_host = num_gpus

    # Data Preprocessing
    print("Download pre-trained model....")
    subprocess.check_call("mkdir -p /opt/ml/code/data/pretrained-models", shell=True)
    subprocess.check_call("wget http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz", shell=True)
    subprocess.check_call("cp ImageNet-R50-AlignPadding.npz data/pretrained-models", shell=True)
    print("Loading data from s3......")
    subprocess.check_call("aws s3 cp s3://armand-ajay-workshop/mask-rcnn/sagemaker/input/train /opt/ml/code/data --recursive --quiet", shell=True)
    print("Loading data finsihed...Install tensorpack....")
    subprocess.check_call("git clone https://github.com/armandmcqueen/tensorpack-mask-rcnn /opt/ml/code/tensorpack-mask-rcnn", shell=True)
    subprocess.check_call("chmod -R +w /opt/ml/code/tensorpack-mask-rcnn", shell=True)
    subprocess.check_call("pip install --ignore-installed -e /opt/ml/code/tensorpack-mask-rcnn/", shell=True)
    subprocess.check_call("chmod +x /opt/ml/code/run.sh", shell=True)
    print("Tensorpack install finished...")

    _start_ssh_daemon()
    # Remove the conflict MPI setting
    subprocess.check_call("sed -ie \"s/btl_tcp_if_exclude/#btl_tcp_if_exclude/g\" /usr/local/etc/openmpi-mca-params.conf", shell=True)

    if current_host == hosts[0]:
        host_list = hosts if process_slots_per_host == 1 else \
            [host + ':{}'.format(process_slots_per_host) for host in hosts]

        num_processes = process_slots_per_host * len(hosts)
        credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
        # Build mpirun file
        mpi_command = [    '#!/usr/bin/env bash \n',
                           '/usr/local/bin/mpirun --allow-run-as-root --display-map --tag-output --host {} \\\n'.format(",".join(host_list)), \
                           ' --mca plm_rsh_no_tree_spawn 1 \\\n', \
                           ' -mca pml ob1 \\\n', \
                           ' -mca btl ^openib \\\n', \
                           ' -bind-to None \\\n', \
                           ' -map-by slot \\\n', \
                           ' -mca btl_vader_single_copy_mechanism none \\\n'
                           ' -mca btl_tcp_if_include {} \\\n'.format(env.network_interface_name), \
                           ' -mca oob_tcp_if_include {} \\\n'.format(env.network_interface_name), \
                           ' -x NCCL_SOCKET_IFNAME={} \\\n'.format(env.network_interface_name), \
                           ' -x NCCL_MIN_NRINGS=8 \\\n', \
                           ' -x HOROVOD_CYCLE_TIME=0.5 \\\n', \
                           ' -x HOROVOD_FUSION_THRESHOLD=67108864 \\\n', \
                           ' -x TENSORPACK_FP16=1 \\\n', \
                           ' -x PATH \\\n', \
                           ' -x LD_LIBRARY_PATH \\\n', \
                           ' -x NCCL_DEBUG=INFO \\\n', \
                           ' -mca orte_abort_on_non_zero_status 1 \\\n', \
                           ' -np {} \\\n'.format(num_processes)]
        for v in credential_vars:
            if v in os.environ:
                mpi_command.append(" -x {} \\\n".format(v))
        for cmd in custom_mpi_cmds:
            mpi_command.append("{} \\\n".format(cmd))
        mpi_command.append("/opt/ml/code/run.sh")
        # Write file and lanch mpi
        with open('mpi_cmd.sh', 'a') as the_file:
            for item in mpi_command: the_file.write(item)
        with open('mpi_cmd.sh', 'r') as the_file:
            logger.info('MPI script:\n\n%s', the_file.read())
        subprocess.check_call("chmod +x mpi_cmd.sh", shell=True)
        _wait_for_worker_nodes_to_start_sshd(hosts)
        subprocess.check_call("./mpi_cmd.sh", shell=True)
    else:
        _wait_master_to_start(hosts[0])
        _wait_master_to_finish(hosts[0])

if __name__ == '__main__':
    hyperparameters = json.loads(os.environ['SM_HPS'])
    current_host = os.environ['SM_CURRENT_HOST']
    hosts = []
    # it may take a while to load all the hosts
    while len(hosts) < hyperparameters["num_nodes"]:
        try:
            hosts = json.loads(os.environ['SM_HOSTS'])
        except: pass
    train(sorted(hosts), current_host, hyperparameters["num_gpus"], hyperparameters["custom_mpi_cmds"])
