# Overview: Files

Currently, there are several files inside this `deep_learning_container` directory. The following listing serves in order to explain, what file is needed for what purpose:

|File|Purpose|
|----|-------|
|[`Dockerfile`](./Dockerfile)|The main file which incorporates all commands in order to define the structure and software stack of the docker container.|
|[`NvidiaNCCL.deb`](./)|NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs which enables fast computation of tensor operations.|
|[`convert_container.sh`](convert_container.sh)|This script converts the docker image of the local docker deamon into a container file representation which the AI System at LRZ is able to read (.sqsh file format for enroot containers). Attention: This only works if the docker_build.sh script has been executed before.|
|[`docker_build.sh`](docker_build.sh)|Run this script in order to create a local docker image out of the Dockerfile.|
|[`docker_run.sh`](docker_run.sh)|Run script in order to start the docker container. Attenion: This only works if the docker_build.sh script has been executed before.|
|[`entrypoint.sh`](entrypoint.sh)|This file is part of the docker container itself. It gets invoked every time the container gets started.|
|[`requirements.txt`](requirements.txt)|This file gets copied into the docker container. It serves the purpose of being the python library database.|
|[`script.sbatch`](script.sbatch)|This script serves as a template, in order to know how to submit a batch job into the SLURM pipeline on the AI System at LRZ.|