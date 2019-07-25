# Building the Wheel

Use the codebase here to build a TF wheel: https://github.com/samikama/tensorflow/commits/GenerateProposalsOp

Apply the diff patches above.



## Building Tensorflow

Requires custom Tensorflow for GPU optimized ops. Build steps were run on the AWS DLAMI 21.2.

```
source activate tensorflow_p36
pip uninstall -y tensorflow horovod

############################################################################################################
# Upgrade Bazel
############################################################################################################ 
rm /home/ubuntu/anaconda3/envs/tensorflow_p36/bin/bazel
wget https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-installer-linux-x86_64.sh
chmod +x bazel-0.19.2-installer-linux-x86_64.sh
./bazel-0.19.2-installer-linux-x86_64.sh --user


############################################################################################################
# Build TF 1.13 with CUDA 10
############################################################################################################

./configure

# XLA JIT: N
# CUDA: Y
# CUDA/CUDNN/NCCL dir: /usr/local/cuda-10.0
# CUDNN: 7.4.1
# NCCL: 2.3.7


############################################################################################################
# Create pip wheel
############################################################################################################

bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg
```


