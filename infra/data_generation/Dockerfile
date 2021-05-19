FROM nvcr.io/nvidia/cuda:11.3.0-devel-ubuntu18.04

LABEL description="This container incorporates everything which is needed for the data generation pipeline"

# Install python and pip package manager
RUN apt update
RUN apt install -y python3-pip python3-dev
RUN pip3 install --upgrade pip

# Install GIT
RUN apt -y install git

# Install Cmake
RUN apt -y install cmake

# Install predefined python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Define entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]