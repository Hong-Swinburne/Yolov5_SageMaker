#!/bin/sh

image=$1
docker run -v $(pwd)/test_dir:/opt/ml--rm ${image} predict
#nvidia-docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} predict