FROM ubuntu:bionic

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN pip3 install numpy
RUN pip3 install plotly
RUN pip3 install torch
RUN pip3 install pytorch3d

# Maybe add kaolin repo to our git and copy it from there to the image
RUN apt-get -y install git
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
RUN cd /kaolin && git checkout v0.9.0
RUN cd /kaolin && python3 setup.py develop