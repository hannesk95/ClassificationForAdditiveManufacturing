#!/bin/sh

# Build docker container image
docker build -t "di-lab" .

# Run container image
docker run --rm --hostname DI-LAB di-lab
