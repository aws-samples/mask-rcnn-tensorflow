FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.12.0-gpu-py310-cu118-ubuntu20.04-ec2

RUN pip3 install --upgrade pip
RUN pip3 uninstall -y tensorflow-io

RUN pip3 install nvidia-cudnn-cu11==8.9.2.26
RUN pip3 install tensorflow-io==0.32.0

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY . /app

RUN pip3 install -e /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser

RUN pip3 install jupyterlab
RUN pip3 install notebook

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "MaskRCNN/train.py"]