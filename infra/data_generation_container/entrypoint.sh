#!/bin/bash

# Clone current status of project repository
git clone https://hannesk95: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@github.com/semester-project/TUM-DI-LAB.git

# Clone and compile binaries for CUDA voxelizer
git clone https://github.com/kctess5/voxelizer.git
cd voxelizer
mkdir build 
cd build
cmake .. 
make
cd /

# Start bash
/bin/bash