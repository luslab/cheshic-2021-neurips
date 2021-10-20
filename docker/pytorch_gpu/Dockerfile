# ARG BASE_CONTAINER=pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
ARG BASE_CONTAINER=pytorch/pytorch:latest
FROM $BASE_CONTAINER

LABEL authors="Chris Cheshire" \
      description="Docker image for running pytorch on GPU CAMP"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV NUMBA_CACHE_DIR='/tmp'

# Install conda packages
COPY environment.yml /
RUN conda env create -f /environment.yml && conda clean -a
ENV PATH /opt/conda/envs/packages/bin:$PATH