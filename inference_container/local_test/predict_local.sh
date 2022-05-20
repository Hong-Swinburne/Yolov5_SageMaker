#!/bin/sh

image=$1
docker run -v $(pwd)/test_dir:/opt/ml/model -it --shm-size 20g --rm ${image} predict
#nvidia-docker run -v $(pwd)/test_dir:/opt/ml/model -it --shm-size 20g --rm ${image} predict