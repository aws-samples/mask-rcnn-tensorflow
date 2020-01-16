# NGC based docker container with custom TF ops compiled for Intel Sandy Bridge and Nvidia V100
# built on base aws maskrcnn image with prebuilt tensorflow
FROM awssamples/mask-rcnn-tensorflow:base

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# add mask-rcnn packages
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev awscli && \
    pip install opencv-python==4.1.1.26

RUN pip uninstall -y pycocotools && \
    pip install pybind11 && \
    pip install scikit-image

RUN wget https://github.com/aws-samples/mask-rcnn-tensorflow/releases/download/v0.0.0/example_log.tar.gz && \
    tar -xzf example_log.tar.gz example_log


# add custom nvidia coco tools
# need to be modified for pybind11 header files
RUN git clone https://github.com/NVIDIA/cocoapi && \
    cd cocoapi/PythonAPI && \
    awk 'NR==1 {$0="#include <python3.6/pybind11/pybind11.h>"} { print }' pycocotools/ext.cpp > pycocotools/ext1.cpp && \
    awk 'NR==2 {$0="#include <python3.6/pybind11/numpy.h>"} { print }' pycocotools/ext1.cpp > pycocotools/ext2.cpp && \
    awk 'NR==3 {$0="#include <python3.6/pybind11/stl.h>"} { print }' pycocotools/ext2.cpp > pycocotools/ext3.cpp && \
    rm pycocotools/ext.cpp && \
    rm pycocotools/ext1.cpp && \
    rm pycocotools/ext2.cpp && \
    mv pycocotools/ext3.cpp pycocotools/ext.cpp && \
    make install


WORKDIR /

# clone repo for mask r-cnn scripts and demos
RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install --ignore-installed -e /mask-rcnn-tensorflow/

RUN apt update && \
    apt upgrade -y && \
    apt install -y openssh-server

RUN mkdir -p /var/run/sshd

RUN pip uninstall -y numpy
RUN pip uninstall -y numpy

RUN pip install numpy==1.16.2