# NGC based docker container with custom TF ops compiled for Intel Sandy Bridge and Nvidia V100
# base image from Nvidia Tensowflow docker image
FROM nvcr.io/nvidia/tensorflow:19.09-py3

WORKDIR /opt/tensorflow

# download patch for custom tensorflow functions
RUN cd tensorflow-source && \
	wget https://github.com/aws-samples/mask-rcnn-tensorflow/releases/download/v0.0.0/SizeFix.patch && \
	patch -p1 < SizeFix.patch && \
	cd ..

# modify nvidia build script to optimize for P3 instances
RUN awk 'NR==59 {$0="export TF_CUDA_COMPUTE_CAPABILITIES=\"7.0\""} { print }' nvbuild.sh > nvbuild_1.sh && \
	awk 'NR==62 {$0="export CC_OPT_FLAGS=\"-march=native\""} { print }' nvbuild_1.sh > nvbuild_new.sh && \
	rm nvbuild_1.sh

# run tensorflow build
RUN chmod +x nvbuild_new.sh
RUN ./nvbuild_new.sh --python3.6

# add mask-rcnn packages
RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    pip install opencv-python

RUN pip uninstall -y pycocotools && \
    pip install pybind11

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

RUN pip uninstall -y numpy

RUN pip install --ignore-installed numpy==1.16.2

WORKDIR /

# clone repo for mask r-cnn scripts and demos
RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install --ignore-installed -e /mask-rcnn-tensorflow/

RUN apt update && \
    apt upgrade -y && \
    apt install -y openssh-server

RUN mkdir -p /var/run/sshd
