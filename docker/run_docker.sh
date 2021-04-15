#!/bin/bash
docker run \
    -itd --rm\
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/deep_learning_from_scratch:/workdir \
    yamamura/cupy-rocm:v9.0.1rc1 
