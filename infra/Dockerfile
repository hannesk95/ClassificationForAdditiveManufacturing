# FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM ubuntu:latest

LABEL description="This container incorporates the entire environment in order to deploy it on LRZ AI System"

# Install python and pip package manager
RUN apt update \
  && apt install -y python3-pip python3-dev \
  && pip3 install --upgrade pip

# Install predefined python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install GIT
RUN apt -y install git

# Install Horovod for GPU parallelization
# RUN HOROVOD_GPU_OPERATIONS=NCCL pip3 install horovod

# Install PyTorch libraries
# RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
# RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
# RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
# RUN pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html

# Change Work Directory
WORKDIR /root/

# Define entrypoint
COPY entrypoint.sh /entrypoint.sh
#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["/entrypoint.sh"]
