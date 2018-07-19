#!/bin/bash

# TODO(minqiyang): move these hack lines to envs
CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PADDLE_PATH=$CURRENT_FILE_DIR/../../Paddle
PADDLE_BUILD_PATH=$PADDLE_PATH/build/add_remote_initer
PADDLE_BENCHMARK_PATH=$PADDLE_PATH/benchmark/fluid

# Get paddle latest commit id
PADDLE_COMMIT=`cd $PADDLE_PATH && git rev-parse --verify HEAD`
echo "Detect PADDLE_PATH:", $PADDLE_PATH, "PADDLE_COMMIT:", $PADDLE_COMMIT

# Prepare the docker image
PADDLE_CLOUD_DOCKER_SERVER=registry.baidu.com
PADDLE_LOCAL_DOCKER_HUB_TAG=minqiyang/ce_dist_resnet50:${PADDLE_COMMIT}
PADDLE_CLOUD_DOCKER_HUB_TAG=${PADDLE_CLOUD_DOCKER_SERVER}/${PADDLE_LOCAL_DOCKER_HUB_TAG}
PADDLE_CLOUD_DOCKER_HUB_USERNAME=minqiyang
PADDLE_CLOUD_DOCKER_HUB_PASSWORD=n5WY5lWculWb

# Copy whl package to local image dir
LOCAL_IMAGE_DIR=./paddle_cloud_docker_job/
cp $PADDLE_BUILD_PATH/python/dist/*.whl $LOCAL_IMAGE_DIR

# Add fluid benchmark code to local image dir
cp -r $PADDLE_BENCHMARK_PATH $CURRENT_FILE_DIR
PADDLE_CLOUD_FILES_PATH=$CURRENT_FILE_DIR/fluid

# Build docker image
docker login -u $PADDLE_CLOUD_DOCKER_HUB_USERNAME -p $PADDLE_CLOUD_DOCKER_HUB_PASSWORD $PADDLE_CLOUD_DOCKER_SERVER
docker build -t $PADDLE_CLOUD_DOCKER_HUB_TAG $LOCAL_IMAGE_DIR
docker push $PADDLE_CLOUD_DOCKER_HUB_TAG

# Submit job to paddle cloud
PADDLE_CLOUD_SERVER=paddlecloud.baidu-int.com
PADDLE_CLOUD_PORT=80
PADDLE_CLOUD_AK=4a9ae3ff4c545dca803a78f15396b0d3
PADDLE_CLOUD_SK=c60a143628ef560aadc96d3b4fd0e8e4
PADDLE_CLOUD_CLUSTER_NAME=paddle-jpaas-ai00-gpu
PADDLE_CLOUD_JOB_VERSION=custom-fluid
PADDLE_CLOUD_JOB_PRIORITY=high
PADDLE_CLOUD_JOB_WALL_TIME=1:00:00
PADDLE_CLOUD_JOB_NAME="ce_dist_resnet50_"$PADDLE_COMMIT

# kpi task custom configuration
PADDLE_CLOUD_PSERVER_COUNT=2
PADDLE_CLOUD_TRAINER_COUNT=2
PADDLE_CLOUD_PSERVER_CPUS=10
PADDLE_CLOUD_TRAINER_CPUS=10
PADDLE_CLOUD_TRAINER_GPUS=2
PADDLE_CLOUD_TRAINER_GPU_TYPE=baidu/gpu_p40
PADDLE_CLOUD_PSERVER_MEM=5Gi
PADDLE_CLOUD_TRAINER_MEM=10Gi

PADDLE_CLOUD_BENCHMARK_FILE="$(python get_benchmark_files.py $PADDLE_CLOUD_FILES_PATH)"
echo "Collect files:", $PADDLE_CLOUD_BENCHMARK_FILE

cd $PADDLE_CLOUD_CLI_PATH && python setup.py install && cd $CURRENT_FILE_DIR

paddlecloud job  --server ${PADDLE_CLOUD_SERVER} \
--port ${PADDLE_CLOUD_PORT} \
--user-ak ${PADDLE_CLOUD_AK} \
--user-sk ${PADDLE_CLOUD_SK} \
train --cluster-name ${PADDLE_CLOUD_CLUSTER_NAME} \
--job-version ${PADDLE_CLOUD_JOB_VERSION} \
--k8s-priority ${PADDLE_CLOUD_JOB_PRIORITY} \
--k8s-wall-time ${PADDLE_CLOUD_JOB_WALL_TIME} \
--job-name ${PADDLE_CLOUD_JOB_WALL_TIME} \
--start-cmd "python fluid/fluid_benchmark.py --model resnet --sync_mode --data_set flowers --iterations 20 --device GPU --gpus 8 --batch_size 32 --pass_num 1000 --update_method pserver" \
--job-conf job_conf.py \
--files ${PADDLE_CLOUD_BENCHMARK_FILE} \
--k8s-not-local \
--k8s-trainers ${PADDLE_CLOUD_TRAINER_COUNT} \
--k8s-cpu-cores ${PADDLE_CLOUD_TRAINER_CPUS} \
--k8s-memory ${PADDLE_CLOUD_TRAINER_MEM} \
--k8s-gpu-type ${PADDLE_CLOUD_TRAINER_GPU_TYPE} \
--k8s-gpu-cards ${PADDLE_CLOUD_TRAINER_GPUS} \
--k8s-ps-num ${PADDLE_CLOUD_PSERVER_COUNT} \
--k8s-ps-cores ${PADDLE_CLOUD_PSERVER_CPUS} \
--k8s-ps-memory ${PADDLE_CLOUD_PSERVER_MEM} \
--image-addr "${PADDLE_CLOUD_DOCKER_HUB_TAG}"

