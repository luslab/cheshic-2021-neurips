ARG BASE_CONTAINER=pytorch/pytorch:latest
FROM $BASE_CONTAINER

# Install conda packages
COPY environment.yml /
RUN conda env create -f /environment.yml && conda clean -a
ENV PATH /opt/conda/envs/packages/bin:$PATH

# Install biggraph
WORKDIR /home
RUN git clone https://github.com/facebookresearch/PyTorch-BigGraph.git
WORKDIR /home/PyTorch-BigGraph
RUN /opt/conda/envs/packages/bin/pip install .
WORKDIR /home