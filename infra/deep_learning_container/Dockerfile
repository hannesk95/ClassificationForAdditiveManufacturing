FROM nvcr.io/nvidia/pytorch:21.04-py3

LABEL description="This container incorporates everything which is needed for the deep learning pipeline"

# Install python and pip package manager
RUN apt update
RUN apt install -y python3-pip python3-dev
RUN pip3 install --upgrade pip

# Install GIT
RUN apt -y install git

# Install Horovod for GPU parallelization
# RUN HOROVOD_GPU_OPERATIONS=NCCL pip3 install horovod

# Install predefined python packages
COPY requirements.txt /tmp/requirements.txt
# RUN pip3 install -r /tmp/requirements.txt

# Define entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]