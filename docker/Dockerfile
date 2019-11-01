FROM tensorflow/tensorflow:1.5.0-gpu-py3
MAINTAINER jchartois@sqli.com
RUN  pip install Augmentor
RUN pip install -U pip numpy
RUN pip3 install opencv-python
RUN pip3 install -U pip numpy
WORKDIR /mnt
RUN apt-get update && apt-get install -y \
    nano \
    python-dev \
    python-pip \
    python3-dev \
    python3-pip \
    python-opencv \
    libopencv-dev \
    libav-tools  \
    libjpeg-dev \
    libpng-dev \
    python-numpy \
    python-pycurl \
    python-opencv
