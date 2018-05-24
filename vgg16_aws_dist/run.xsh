#!/bin/bash

PADDLE_PATH=../../../
paddle_build_path=$PADDLE_PATH/build
paddle_docker_hub_tag="paddlepaddlece/paddle:latest"
vgg16_test_dockerhub_tag="paddlepaddlece/vgg16_dist:latest"
training_command="local:no,batch_size:128,num_passes:1"

# clean up docker
docker system prune -f

# loginto docker hub
docker login -u $DOCKER_HUB_USERNAME -p $DOCKER_HUB_PASSWORD

# create paddle docker image
echo "going to build and push paddle production image"
docker build -t $paddle_docker_hub_tag $paddle_build_path
docker push $paddle_docker_hub_tag

# build test docker image
echo "going to prepare and build vgg16_dist_test"
if [ ! -d vgg16_dist_test ]; then
    echo "No vgg16_dist_test repo found, going to clone one"
    git clone https://github.com/putcn/vgg16_dist_test.git
fi
cd vgg16_dist_test
if [ -d ~/.cache/paddle/dataset/cifar ]; then
    echo "host cifar cache found, copying it to docker root"
    mkdir -p .cache/paddle/dataset/
    cp -r -f ~/.cache/paddle/dataset/cifar .cache/paddle/dataset/
fi
git pull
cd ..
echo "going to build vgg16_dist_test docker image and push it"
docker build -t $vgg16_test_dockerhub_tag ./vgg16_dist_test
docker push $vgg16_test_dockerhub_tag
docker logout

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
python aws_runner/client/ce_runner.py \
    --key_name aws_benchmark_us_east \
    --security_group_id sg-95539dff \
    --online_mode yes \
    --trainer_count 2 \
    --pserver_count 2 \
    --pserver_command $training_command \
    --trainer_command $training_command \
    --docker_image $vgg16_test_dockerhub_tag