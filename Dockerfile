FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN pip3 install -r requirements.txt

RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
RUN pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html
RUN pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${1.8.1}+${11.2.2}.html

# Maybe add kaolin repo to our git and copy it from there to the image
RUN apt-get -y install git
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
RUN cd /kaolin && git checkout v0.9.0
RUN cd /kaolin && python3 setup.py develop

# Same as kaolin repo
RUN git clone https://github.com/PyMesh/PyMesh.git
RUN cd PyMesh && git submodule update --init && export PYMESH_PATH=`pwd`