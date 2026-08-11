# syntax=docker/dockerfile:1

ARG FROM_IMAGE=ubuntu:24.04
FROM ${FROM_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        g++ \
        libomp-dev \
        libopenmpi-dev \
        openmpi-bin \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/jittor-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONIOENCODING=utf8

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /opt/jittor

COPY pyproject.toml setup.py README.md MANIFEST.in LICENSE.txt ./
COPY requirements/examples.txt ./requirements/examples.txt
COPY python ./python
COPY examples ./examples

RUN python -m pip install --no-cache-dir . \
    && python -m pip install --no-cache-dir -r requirements/examples.txt \
    && find examples/notebooks -type f -name '*.md' ! -name README.md -print0 \
       | xargs -0 -n1 python -m jupytext --to ipynb \
    && nvcc_path= python -m jittor.selftest

CMD ["python", "-m", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--ServerApp.root_dir=/opt/jittor/examples/notebooks"]
