#!/bin/bash

set -xe

CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PADDLE_PATH=$CURRENT_FILE_DIR/../../..
paddle_build_path=$PADDLE_PATH/build
paddle_docker_hub_tag="paddlepaddlece/paddle:latest"
fluid_benchmark_dockerhub_tag="paddlepaddlece/fluid_benchmark:latest"
training_command="update_method:pserver,acc_target:0.6,iterations:100,pass_num:1"

# clean up docker
# docker system prune -f

# loginto docker hub
# login is now performed in teamcity
# docker login -u $DOCKER_HUB_USERNAME -p $DOCKER_HUB_PASSWORD

# create paddle docker image
echo "going to build and push paddle production image"
docker build -t $paddle_docker_hub_tag $paddle_build_path
docker push $paddle_docker_hub_tag

# build test docker image
cd $CURRENT_FILE_DIR

cd fluid_benchmark_for_aws
if [ -d ~/.cache/paddle/dataset/cifar ]; then
    echo "host cifar dataset cache found, copying it to docker root"
    mkdir -p .cache/paddle/dataset/
    cp -r -f ~/.cache/paddle/dataset/cifar .cache/paddle/dataset/
fi

if [ -d ~/.cache/paddle/dataset/flowers ]; then
    echo "host flower dataset cache found, copying it to docker root"
    mkdir -p .cache/paddle/dataset/
    cp -r -f ~/.cache/paddle/dataset/flowers .cache/paddle/dataset/
fi

cd ..

echo "going to build fluid_benchmark_for_aws docker image and push it"
docker build -t $fluid_benchmark_dockerhub_tag ./fluid_benchmark_for_aws
docker push $fluid_benchmark_dockerhub_tag

# fetch runner and install dependencies
echo "going to work with aws_runner"
if [ ! -d aws_runner ]; then
    echo "no aws_runner found, cloning one"
    git clone https://github.com/putcn/aws_runner.git
fi
cd aws_runner
git pull
cd ..
echo "going to install aws_runner dependencies"
pip install -r aws_runner/client/requirements.txt

echo "going to start testing"
# start aws testingr
python ce_runner.py \
    --key_name aws_benchmark_us_east \
    --security_group_id sg-95539dff \
    --online_mode yes \
    --pserver_command $training_command \
    --trainer_command $training_command \
    --docker_image $fluid_benchmark_dockerhub_tag
