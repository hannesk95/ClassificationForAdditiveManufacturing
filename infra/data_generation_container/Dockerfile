FROM nvcr.io/nvidia/cuda:11.3.0-devel-ubuntu18.04

LABEL description="This container incorporates everything which is needed for the data generation pipeline"

# Install python and pip package manager
RUN apt update
RUN apt install -y python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN apt install wget

# Install GIT
RUN apt -y install git

# Install Cmake
RUN apt -y install cmake

# Install predefined python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install Open3D
RUN wget https://github.com/intel-isl/Open3D/releases/download/v0.12.0/open3d-app-0.12.0-Ubuntu_20.04.deb
RUN apt install libc++1 libpng16-16 libglfw3
RUN dpkg -i open3d-app-0.12.0-Ubuntu_20.04.deb

# Define entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]