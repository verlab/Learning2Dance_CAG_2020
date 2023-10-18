#FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Fixing broken key
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt upgrade -y
RUN apt install ffmpeg pkg-config libfreetype6-dev libhdf5-serial-dev libxslt1-dev -y
# OpenCV needed
RUN apt install libsm6 libxrender1 libfontconfig1 -y

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

HEALTHCHECK --timeout=1m --retries=1 \
    CMD python3 -c "import torch"