# DockerHub unaltered mirror of AWS Deep Learning Container
FROM 578276202366.dkr.ecr.us-east-1.amazonaws.com/dlami

RUN apt-get install less

# Need to reinstall some libraries the DL container provides due to custom Tensorflow binary
RUN pip uninstall -y tensorflow tensorboard tensorflow-estimator keras h5py horovod numpy

# Download and install custom Tensorflow binary
RUN wget https://github.com/aws-samples/mask-rcnn-tensorflow/releases/download/v0.0.0/tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-estimator==1.13.0 && \
    rm tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl

RUN pip install keras h5py

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1  pip install --no-cache-dir horovod==0.15.2 && \
    ldconfig


# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config


RUN pip install Cython
RUN pip install ujson opencv-python matplotlib
RUN pip install --ignore-installed numpy==1.16.2
RUN pip install pybind11

# add custom nvidia coco tools
# need to be modified for pybind11 header files
RUN git clone https://github.com/NVIDIA/cocoapi && \
    cd cocoapi/PythonAPI && \
    make install

RUN pip install mpi4py
RUN pip install numba

# For Sagemaker
RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY run_mpi.py /opt/ml/code/run_mpi.py
COPY run.sh /opt/ml/code/run.sh
RUN chmod +x /opt/ml/code/run.sh

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow /opt/ml/code/mask-rcnn-tensorflow
RUN chmod -R +w  /opt/ml/code/mask-rcnn-tensorflow
RUN pip install --ignore-installed -e  /opt/ml/code/mask-rcnn-tensorflow

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM run_mpi.py
