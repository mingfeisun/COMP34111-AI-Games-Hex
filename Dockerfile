FROM ubuntu:22.04

ENV HOME="/home/hex"
ARG UID
RUN useradd -u $UID --create-home hex

RUN apt-get update && apt-get install
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y default-jre

# install requirements
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install numpy pandas scikit-learn==1.5.2 tensorflow==2.16.1 keras==3.6.0 torch==2.4.1

USER root
WORKDIR /home/hex
