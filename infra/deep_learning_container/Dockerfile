FROM nvcr.io/nvidia/pytorch:21.04-py3

LABEL description="This container incorporates everything which is needed for the deep learning pipeline"

ARG DEBIAN_FRONTEND=noninteractive

# Install python and pip package manager
RUN apt update
RUN apt install -y python3-pip python3-dev
RUN pip3 install --upgrade pip

# Install GIT
RUN apt -y install git

# Install Cmake
RUN apt -y install cmake

# Install predefined python packages
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Install NCCL and horovod for GPU parallelization
COPY NvidiaNCCL.deb NvidiaNCCL.deb
RUN dpkg -i NvidiaNCCL.deb
RUN apt update
RUN apt install libnccl2 libnccl-dev
RUN HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

# Install screen
RUN apt -y install screen

# Copy .sbatch script into container
COPY script.sbatch script.sbatch

# Define entrypoint
COPY entrypoint.sh entrypoint.sh
ENTRYPOINT ["bash","entrypoint.sh"]