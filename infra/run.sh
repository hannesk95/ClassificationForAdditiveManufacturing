#!/bin/sh

# Build docker container image
docker build -t tum-di-lab .

# Run container image
docker run -it --rm tum-di-lab
