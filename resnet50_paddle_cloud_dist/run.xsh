#!/bin/bash

# TODO(minqiyang): move these hack lines to envs
CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PADDLE_PATH=$CURRENT_FILE_DIR/../../..
PADDLE_BUILD_PATH=$PADDLE_PATH/build
PADDLE_BENCHMARK_PATH=$PADDLE_PATH/benchmark/fluid
PADDLE_CLOUD_FILES_PATH=./fluid

# 1. Get paddle latest commit id
PADDLE_COMMIT=`cd $PADDLE_PATH && git rev-parse --verify --short HEAD`
echo "Detect PADDLE_PATH:", $PADDLE_PATH, "PADDLE_COMMIT:", $PADDLE_COMMIT
export PADDLE_CLOUD_JOB_NAME="`basename $CURRENT_FILE_DIR`_"$PADDLE_COMMIT

# 2. Set all cloud helper env
PADDLE_CLOUD_HELPER_PATH=paddle-dist-ce-helper
PADDLE_CLOUD_HELPER_GIT_SERVER=ssh://minqiyang@icode.baidu.com:8235/baidu/paddle-ce/paddle-dist-ce-helper
PADDLE_CLOUD_CLI_PATH=$CURRENT_FILE_DIR/paddle_cloud_cli/
LOCAL_IMAGE_DIR=$CURRENT_FILE_DIR/$PADDLE_CLOUD_HELPER_PATH/paddle_cloud_docker_job/

# 3. Clone helper and source all envs
rm -rf $PADDLE_CLOUD_HELPER_PATH && git clone $PADDLE_CLOUD_HELPER_GIT_SERVER $PADDLE_CLOUD_HELPER_PATH && cd $PADDLE_CLOUD_HELPER_PATH && source envs.sh && cd $CURRENT_FILE_DIR

# 4. Prepare the docker image
PADDLE_LOCAL_DOCKER_HUB_TAG=minqiyang/ce_dist_resnet50:${PADDLE_COMMIT}
PADDLE_CLOUD_DOCKER_HUB_TAG=${PADDLE_CLOUD_DOCKER_SERVER}/${PADDLE_LOCAL_DOCKER_HUB_TAG}

# 5. Copy whl package to local image dir
PADDLE_CODE_HOST_PATH=/home/teamcity/system/git
PADDLE_CODE_PATH=paddle_dist_ce_${PADDLE_COMMIT}
docker run --privileged --net=host -v  ${PADDLE_CODE_HOST_PATH}:/paddle -e "WITH_ANAKIN=OFF" -e "WITH_DOC=OFF" -e "WITH_GPU=ON" -e "WITH_DISTRIBUTE=ON" -e "WITH_SWIG_PY=ON" -e "WITH_PYTHON=ON" -e "PYTHON_ABI=cp27-cp27mu" -e "https_proxy=http://172.19.61.250:3128" -e "PADDLE_COMMIT=${PADDLE_COMMIT}" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -c "cd /paddle && rm -rf ${PADDLE_CODE_PATH}* && mkdir ${PADDLE_CODE_PATH} && cd ${PADDLE_CODE_PATH} && git clone https://github.com/PaddlePaddle/Paddle && cd Paddle && ./paddle/scripts/paddle_build.sh build > /dev/null" && rm -rf $LOCAL_IMAGE_DIR/*.whl && cp $PADDLE_CODE_HOST_PATH/$PADDLE_CODE_PATH/Paddle/build/python/dist/*.whl $LOCAL_IMAGE_DIR && rm -rf $PADDLE_CODE_HOST_PATH/$PADDLE_CODE_PATH

# 6. Build docker image
docker login -u $PADDLE_CLOUD_DOCKER_HUB_USERNAME -p $PADDLE_CLOUD_DOCKER_HUB_PASSWORD -e minqiyang@baidu.com $PADDLE_CLOUD_DOCKER_SERVER
docker build -t $PADDLE_CLOUD_DOCKER_HUB_TAG $LOCAL_IMAGE_DIR
docker push $PADDLE_CLOUD_DOCKER_HUB_TAG

# 7. Submit job to paddle cloud
PADDLE_CLOUD_JOB_VERSION=paddle-fluid-custom
PADDLE_CLOUD_JOB_PRIORITY=high
PADDLE_CLOUD_JOB_WALL_TIME=1:00:00

# kpi task custom configuration
PADDLE_CLOUD_PSERVER_COUNT=2
PADDLE_CLOUD_TRAINER_COUNT=2
PADDLE_CLOUD_PSERVER_CPUS=10
PADDLE_CLOUD_TRAINER_CPUS=4
PADDLE_CLOUD_TRAINER_GPUS=2
PADDLE_CLOUD_TRAINER_GPU_TYPE=baidu/gpu_p40
PADDLE_CLOUD_PSERVER_MEM=10Gi
PADDLE_CLOUD_TRAINER_MEM=100Gi

PADDLE_CLOUD_BENCHMARK_FILE="$(python get_benchmark_files.py $PADDLE_CLOUD_FILES_PATH)"
echo "Collect files:", $PADDLE_CLOUD_BENCHMARK_FILE

rm -rf $PADDLE_CLOUD_CLI_PATH && mkdir $PADDLE_CLOUD_CLI_PATH

wget $PADDLE_CLOUD_CLI_SERVER -O $PADDLE_CLOUD_CLI_PATH/$PADDLE_CLOUD_CLI_PKG_NAME && cd $PADDLE_CLOUD_CLI_PATH && tar xzf $PADDLE_CLOUD_CLI_PKG_NAME && cd $PADDLE_CLOUD_CLI_VERSION && python setup.py install && cd $CURRENT_FILE_DIR

echo "Installed paddle cloud cli:", $PADDLE_CLOUD_CLI_PATH/$PADDLE_CLOUD_CLI_PKG_NAME

cat <<EOF
Submit PaddleCloud job: paddlecloud job  --server ${PADDLE_CLOUD_SERVER}
--port ${PADDLE_CLOUD_PORT}
--user-ak ${PADDLE_CLOUD_AK}
--user-sk ${PADDLE_CLOUD_SK}
train --cluster-name ${PADDLE_CLOUD_CLUSTER_NAME}
--job-version ${PADDLE_CLOUD_JOB_VERSION}
--k8s-priority ${PADDLE_CLOUD_JOB_PRIORITY}
--k8s-wall-time ${PADDLE_CLOUD_JOB_WALL_TIME}
--job-name ${PADDLE_CLOUD_JOB_NAME}
--start-cmd "GLOG_logtostderr=1 GLOG_v=4 python fluid_benchmark.py --model resnet --data_set flowers --iterations 20 --device GPU --gpus ${PADDLE_CLOUD_TRAINER_GPUS} --batch_size 32 --pass_num 50 --update_method pserver --no_random --no_split_var"
--job-conf $PADDLE_CLOUD_HELPER_PATH/job_conf.py
--files ${PADDLE_CLOUD_BENCHMARK_FILE}
--k8s-not-local
--k8s-trainers ${PADDLE_CLOUD_TRAINER_COUNT}
--k8s-cpu-cores ${PADDLE_CLOUD_TRAINER_CPUS}
--k8s-memory ${PADDLE_CLOUD_TRAINER_MEM}
--k8s-gpu-type ${PADDLE_CLOUD_TRAINER_GPU_TYPE}
--k8s-gpu-cards ${PADDLE_CLOUD_TRAINER_GPUS}
--k8s-ps-num ${PADDLE_CLOUD_PSERVER_COUNT}
--k8s-ps-cores ${PADDLE_CLOUD_PSERVER_CPUS}
--k8s-ps-memory ${PADDLE_CLOUD_PSERVER_MEM}
--image-addr "${PADDLE_CLOUD_DOCKER_HUB_TAG}"
EOF

#TODO(minqiyang):
# 1. Fix the continuous increase of activate thread count
# 2. Fix the no split var caused by ir
PADDLE_CLOUD_RESULT=$(paddlecloud job --server ${PADDLE_CLOUD_SERVER} \
--port ${PADDLE_CLOUD_PORT} \
--user-ak ${PADDLE_CLOUD_AK} \
--user-sk ${PADDLE_CLOUD_SK} \
train --cluster-name ${PADDLE_CLOUD_CLUSTER_NAME} \
--job-version ${PADDLE_CLOUD_JOB_VERSION} \
--k8s-priority ${PADDLE_CLOUD_JOB_PRIORITY} \
--k8s-wall-time ${PADDLE_CLOUD_JOB_WALL_TIME} \
--job-name ${PADDLE_CLOUD_JOB_NAME} \
--start-cmd "GLOG_logtostderr=1 GLOG_v=4 python fluid_benchmark.py --model resnet --data_set flowers --iterations 20 --device GPU --gpus ${PADDLE_CLOUD_TRAINER_GPUS} --batch_size 32 --pass_num 50 --update_method pserver --no_random --no_split_var" \
--job-conf $PADDLE_CLOUD_HELPER_PATH/job_conf.py \
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
--image-addr "${PADDLE_CLOUD_DOCKER_HUB_TAG}")

PADDLE_CLOUD_RET_CODE=$?

if [ $PADDLE_CLOUD_RET_CODE -ne 0 ]; then
  echo "Submit to PaddleCloud failed, ret code: " $PADDLE_CLOUD_RET_CODE ", errmsg: " $PADDLE_CLOUD_RESULT
  exit 1
fi

echo "PaddleCloud submit result:" $PADDLE_CLOUD_RESULT " ret code: " $PADDLE_CLOUD_RET_CODE

# 8. Collect paddle cloud job result
python get_paddle_cloud_job_info.py $PADDLE_CLOUD_RET_CODE "$PADDLE_CLOUD_RESULT"

PADDLE_CLOUD_JOB_INFO_CODE=$?

PADDLE_CLOUD_JOB_ID=$(cat ./paddle_cloud_job_id)

echo ${HADOOP_PATH}/${PADDLE_CLOUD_USER_ID}/${PADDLE_CLOUD_JOB_ID}/rank-00000/training_result
if [ $PADDLE_CLOUD_JOB_INFO_CODE -eq 0 ]; then
  rm -rf training_result* && cd $PADDLE_CLOUD_HELPER_PATH && ${HADOOP_BIN} fs -Dhadoop.job.ugi=${HADOOP_UGI} -fs ${HADOOP_FS} -get ${HADOOP_PATH}/${PADDLE_CLOUD_USER_ID}/${PADDLE_CLOUD_JOB_ID}/rank-00000/training_result ./ && mv training_result ../ && cd ..
  python record_kpi.py ./training_result
else
  exit 1
fi
