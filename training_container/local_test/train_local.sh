#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

nvidia-docker run -v $(pwd)/test_dir:/opt/ml -it --shm-size=20g --rm ${image} train
# docker run -v $(pwd)/test_dir:/opt/ml -it --shm-size=10g --rm ${image} /bin/bash train
