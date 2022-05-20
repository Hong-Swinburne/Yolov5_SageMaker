#!/bin/bash
username=$1
pwd=$2
image=$3
version=$4
base_img=$5


if [ "$username" == "" ]
then
    echo "Usage: $0 <username in bitbucket>"
    exit 1
fi
if [ "$pwd" == "" ]
then
    echo "Usage: $0 <password>"
    exit 1
fi
if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi
if [ "$version" == "" ]
then
    version = latest
fi
if [ "$base_img" == "" ]
then
    base_img='pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime'
fi

account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${version}"

echo ${fullname}

echo 'base_img:'$base_img

echo "build docker image"

docker build --build-arg repo_username=${username} --build-arg repo_pwd=${pwd} --build-arg BASE_IMG=${base_img} -t ${image} -f Dockerfile .
docker tag ${image} ${fullname}

login ecr
$(aws ecr get-login --no-include-email --registry-ids ${account})
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

# check if the repository exists, if no create a new one with ${image} 
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

echo "push docker image to ecr"

docker push ${fullname}