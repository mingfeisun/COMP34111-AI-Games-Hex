FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

ENV HOME="/home/hex"
# UID default: the build also works where $UID is not set (for example PowerShell)
ARG UID=1000
RUN useradd -u $UID --create-home hex

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    python3.11-dev \
    git \
    curl \
    wget \
    g++\
    && \
    (apt-get install -y openjdk-21-jdk-headless || apt-get install -y openjdk-17-jdk-headless) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11


# install requirements
RUN python3.11 -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1

####### Use if you have GPU and want CUDA support#######
# RUN python3.11 -m pip install --timeout=1000 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

####### Default: the same torch as in the 2025 image #######
# On Linux, torch 2.5.1 from PyPI includes the CUDA libraries. Games run
# without a GPU (no --gpus flag), so torch.cuda.is_available() is False and
# torch uses the CPU. Keep this line unchanged so that last year's bots (the
# reference bots) see the same torch build.
RUN python3.11 -m pip install --timeout=1000 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN python3.11 -m pip install numpy==2.1.3 pandas==2.3.1 scikit-learn==1.7.2

# record the exact versions in this image (compare images with this file)
RUN python3.11 -m pip freeze > /etc/hex-pip-freeze.txt && java -version 2> /etc/hex-java-version.txt


WORKDIR /home/hex
