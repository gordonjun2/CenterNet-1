# NVIDIA CUDA

FROM nvidia/cuda:10.1-cudnn7-devel

# Python 3.6
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.6 python3.6-dev python3-pip wget git sudo nano && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://conda.anaconda.org/conda-forge/linux-64/ca-certificates-2018.4.16-0.tar.bz2 && \
    tar -xjf ca-certificates-2018.4.16-0.tar.bz2 -C /usr/bin && \
    rm ca-certificates-2018.4.16-0.tar.bz2

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

# create a root user
#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER root

ENV PATH="/home/root/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies and packages
RUN apt-get update && apt-get install -y \
    libsm6 libxrender1 libfontconfig1 python3.6-tk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/root

RUN sudo git clone https://github.com/gordonjun2/CenterNet-1.git /home/root/CenterNet-1

WORKDIR /home/root/CenterNet-1

RUN pip install -r requirements.txt

    # Cudatoolkit 10.1
RUN sudo wget https://anaconda.org/anaconda/cudatoolkit/10.1.243/download/linux-64/cudatoolkit-10.1.243-h6bb024c_0.tar.bz2 && \
    sudo tar -xjf cudatoolkit-10.1.243-h6bb024c_0.tar.bz2 -C /usr/bin && \
    sudo rm cudatoolkit-10.1.243-h6bb024c_0.tar.bz2

WORKDIR /home/root/CenterNet-1/src/lib/external

RUN make

WORKDIR /home/root/CenterNet-1/cocoapi/PythonAPI

RUN python3 setup.py build_ext install

# CUDA Setting
ENV FORCE_CUDA="0"
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /home/root/CenterNet-1/src/lib/models/networks/DCNv2

RUN ./make.sh

WORKDIR /home/root/CenterNet-1


