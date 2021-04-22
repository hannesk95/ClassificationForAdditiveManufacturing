#!/bin/sh
# This is the command to be executed when the container starts

# Clone Project Repository
rm -rf TUM-DI-LAB
git clone https://hannesk95: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@github.com/semester-project/TUM-DI-LAB.git

# Clone Kaolin framework
rm -rf kaolin
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
#cd /kaolin && git checkout v0.9.0
#cd /kaolin && python3 setup.py develop

# Clone PyMesh framework
rm -rf PyMesh
git clone https://github.com/PyMesh/PyMesh.git
#cd PyMesh && git submodule update --init && export PYMESH_PATH=`pwd`
/bin/bash