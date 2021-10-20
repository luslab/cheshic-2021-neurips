FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV NUMBA_CACHE_DIR='/tmp'

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

ENV PATH /opt/conda/bin:$PATH

# Reload and update conda
# Conda update base fixes the "module named conda not found" issue
RUN touch ~/.bashrc
RUN conda update -n base conda
RUN conda update --all

# Install conda packages
COPY environment.yml /
RUN conda env create -f /environment.yml && conda clean -a
ENV PATH /opt/conda/envs/packages/bin:$PATH

# # ARG BASE_CONTAINER=pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
# ARG BASE_CONTAINER=pytorch/pytorch:latest
# FROM $BASE_CONTAINER

# LABEL authors="Chris Cheshire" \
#       description="Docker image for running pytorch on GPU CAMP"

# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# ENV NUMBA_CACHE_DIR='/tmp'

# # Install conda packages
# COPY environment.yml /
# RUN conda env create -f /environment.yml && conda clean -a
# ENV PATH /opt/conda/envs/packages/bin:$PATH