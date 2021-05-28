FROM nvcr.io/nvidia/cuda:11.3.0-devel-ubuntu18.04

LABEL description="This container incorporates everything which is needed for the data generation pipeline"

# Install python and pip package manager
RUN apt update
RUN apt install -y python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN apt -y install wget
RUN apt -y install nano

# Install GIT
RUN apt -y install git

# Install Cmake
RUN apt-get -y install apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update
RUN apt-get -y install cmake

# Install predefined python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install Open3D
RUN wget https://github.com/intel-isl/Open3D/releases/download/v0.12.0/open3d-app-0.12.0_Ubuntu-18.04.deb
RUN apt-get -y install ffmpeg libsm6 libxext6
RUN apt-get -y install libc++1 libpng16-16 libglfw3
RUN dpkg -i open3d-app-0.12.0_Ubuntu-18.04.deb

# Install GLU
RUN apt install libgl1-mesa-dev libglu1-mesa-dev
RUN apt install freeglut3-dev

# Copy config.ini file into container
COPY config.ini config.ini

# Define entrypoint
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash","/entrypoint.sh"]