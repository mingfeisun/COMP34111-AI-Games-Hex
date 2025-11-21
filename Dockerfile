FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

ENV HOME="/home/hex"
ARG UID
RUN useradd -u $UID --create-home hex

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    wget \
    default-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11


# install requirements
RUN python3.11 -m pip install --upgrade pip setuptools wheel

####### Use if you have GPU and want CUDA support#######
# RUN python3.11 -m pip install tensorflow[and-cuda]==2.19.0
# RUN python3.11 -m pip install --timeout=1000 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

####### Use if you do not need CUDA support#######
RUN python3.11 -m pip install tensorflow==2.19.0
RUN python3.11 -m pip install --timeout=1000 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN python3.11 -m pip install numpy==2.1.3 pandas==2.3.1 scikit-learn


WORKDIR /home/hex
