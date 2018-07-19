FROM paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5
MAINTAINER PaddlePaddle Authors <paddle-dev@baidu.com>
ENV HOME /root
ADD ./*.whl /
# run paddle version to install python packages first
RUN apt-get update &&        apt-get install -y --allow-downgrades libnccl2=2.1.2-1+cuda8.0 libnccl-dev=2.1.2-1+cuda8.0 &&        apt-get install -y wget python-pip python-opencv libgtk2.0-dev dmidecode python-tk && easy_install -U pip &&         pip install /*.whl; apt-get install -f -y &&         apt-get clean -y &&         rm -f /*.whl &&         paddle version &&         ldconfig
RUN ln -sf /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so.7.1.1 /usr/lib/x86_64-linux-gnu/libcudnn.so
RUN ln -sf /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcublas.so.8.0.88 /usr/lib/x86_64-linux-gnu/libcublas.so
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV NCCL_LAUNCH_MODE PARALLEL
# default command shows the paddle version and exit
CMD ["paddle", "version"]
