FROM paddlepaddlece/paddle:latest

ENV HOME /root
COPY ./ /root/
WORKDIR /root
RUN apt install -y python-opencv
ENTRYPOINT ["python", "vgg16_fluid.py"]
