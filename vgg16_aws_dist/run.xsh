#!/usr/bin/env xonsh
import os

workspace = os.path.dirname(os.path.realpath(__file__)) 
pjoin = os.path.join
normpath = os.path.normpath
paddle_build_path = normpath(pjoin(workspace, '../../build'))
paddle_docker_hub_tag = "paddlepaddlece/paddle:latest"
vgg16_test_dockerhub_tag = "paddlepaddlece/vgg16_dist:latest"

# loginto docker hub
docker login -u $DOCKER_HUB_USERNAME -p $DOCKER_HUB_PASSWORD

# for aws client
AWS_SHARED_CREDENTIALS_FILE = "/workspace/dist_files/.aws"

# create paddle docker image
docker build -t @paddle_docker_hub_tag @(paddle_build_path)
docker push @paddle_docker_hub_tag

# build test docker image
git clone https://github.com/putcn/vgg16_dist_test.git
docker build -t @(vgg16_test_dockerhub_tag) ./vgg16_dist_test
docker push @(vgg16_test_dockerhub_tag)
docker logout

# fetch runner and install dependencies
git clone https://github.com/putcn/aws_runner.git
pip install -r aws_runner/client/requirements.txt

# start aws testing
python aws_runner/client/ce_runner.py \ 
--key_name aws_benchmark_us_east \
--pem_path /workspace/dist_files/aws_benchmark_us_east.pem \
--security_group_id sg-cecaeba5 \
--online_mode yes \
--trainer_count 2 \
--pserver_count 2 \
--pserver_command local:no,batch_size:128,num_passes:1 \
--trainer_command local:no,batch_size:128,num_passes:1 \
--docker_image @(vgg16_test_dockerhub_tag)

