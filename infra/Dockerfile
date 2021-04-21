# FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
FROM ubuntu:latest

LABEL date="21.04.2021"
LABEL description="This container incorporates the entire environment in order to deploy it on LRZ infrastructure"

# Install python and pip package manager
RUN apt update \
  && apt install -y python3-pip python3-dev \
  && pip3 install --upgrade pip

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install GIT
RUN apt-get -y install git

# Install Horovod for GPU parallelization
# RUN HOROVOD_GPU_OPERATIONS=NCCL pip3 install horovod

# Install PyTorch libraries
# RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
# RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
# RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
# RUN pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html

# Clone Kaolin framework
# RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
# RUN cd /kaolin && git checkout v0.9.0
# RUN cd /kaolin && python3 setup.py develop

# Clone PyMesh framework
# RUN git clone https://github.com/PyMesh/PyMesh.git
# RUN cd PyMesh && git submodule update --init && export PYMESH_PATH=`pwd`

# Clone Project Repository
# WORKDIR /root/repository
# RUN git clone https://github.com/semester-project/TUM-DI-LAB.git

# Copy example python script inside the container
WORKDIR /root/python_scripts
COPY ./example.py .

# Define entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
