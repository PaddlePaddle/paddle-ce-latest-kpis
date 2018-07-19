FROM registry.baidu.com/paddlecloud/paddlecloud-runenv-centos6u3-online:paddlecloud-gcc482-cuda8.0_cudnn5
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
USER root
ENV HOME /root
ADD ./*.whl /tmp/
ADD pip.conf /etc/pip.conf
# run paddle version to install python packages first
ENV LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v5/cuda/lib64/:/home/work/cuda-8.0/lib64/:/usr/local/lib/:${LD_LIBRARY_PATH}
RUN pip uninstall -y paddlepaddle && pip --default-timeout=100 install /tmp/*.whl --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com && rm -f /*.whl && paddle version && ldconfig
ENV NCCL_LAUNCH_MODE PARALLEL
# default command shows the paddle version and exit
CMD ["paddle", "version"]
