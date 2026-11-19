# Hex engine image for COMP34111 2026-2027 coursework.
#
# Two targets in this file:
#   game  (default)  the environment in which every game is played and marked:
#                    docker build -t hex .
#   build            game + compilers and toolchains; we use it to rebuild the
#                    compiled parts of a submission with its build.sh:
#                    docker build --target build -t hex-build .
#
# The marking machine is x86-64 Linux. On an Apple Silicon Mac, you MUST add
# --platform linux/amd64 to both commands, or your compiled code will not run.
# No GPU will be available at runtime so PyTorch is installed with CPU-only 
# support (no CUDA).

FROM ubuntu:24.04 AS game

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        g++ make \
        openjdk-21-jdk-headless \
        ca-certificates curl git tzdata \
    && rm -rf /var/lib/apt/lists/*

# User "hex". 
# Games run as root in the container. The engine starts each agent as its own
# user (agent1, agent2), so agents cannot interact except through the game engine.
# UID=1000 is default ubuntu user, remove to prevent clashes when mounting
ARG UID=1000
RUN (userdel -r ubuntu 2>/dev/null || true) && \
    useradd -o -u "$UID" --create-home hex && \
    useradd -u 10001 --no-create-home --shell /usr/sbin/nologin agent1 && \
    useradd -u 10002 --no-create-home --shell /usr/sbin/nologin agent2
ENV HOME="/home/hex"

# Python 3.12 packages installed in venv. Versions are pinned; see
# /etc/hex-pip-freeze.txt in the image for a full list of packages.
RUN python3.12 -m venv "$VIRTUAL_ENV" && \
    pip install --no-cache-dir pip==26.3 setuptools==84.2.1 wheel==0.48.0
RUN pip install --no-cache-dir torch==2.15.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir numpy==2.5.5 pandas==3.1.1 scipy==1.18.1 scikit-learn==1.9.1

# record the exact versions in this image
RUN pip freeze > /etc/hex-pip-freeze.txt && \
    python --version > /etc/hex-python-version.txt && \
    java -version 2> /etc/hex-java-version.txt

WORKDIR /home/hex


FROM game AS build

# C/C++ build tools and Python extension modules (pybind11, Cython)
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11==3.1.0 Cython==3.3.1

# Go
ARG GO_VERSION=1.27.3
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" | tar -C /usr/local -xz
ENV PATH=/usr/local/go/bin:$PATH

# Rust, with the musl target for statically linked binaries
ARG RUST_VERSION=1.100.0
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --default-toolchain "${RUST_VERSION}" && \
    rustup target add x86_64-unknown-linux-musl && \
    chmod -R a+rwX "$RUSTUP_HOME" "$CARGO_HOME"

RUN (go version; rustc --version; cmake --version | head -1) > /etc/hex-build-tools.txt

# You can use anything in statically linked binaries in x86-64 Linux.
# Speak to the course staff if you want to use a different language or toolchain.
# We cannot guarantee that your submission will run if you use a different toolchain.

# Last stage: "docker build -t hex ." builds "game".
FROM game
