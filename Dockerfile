ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable
FROM $BASE_CONTAINER

USER root

# tensorflow, pytorch stable versions
# https://pytorch.org/get-started/previous-versions/
# https://www.tensorflow.org/install/source#linux

RUN apt-get update && \
	apt-get install -y

USER jovyan

# Install pillow<7 due to dependency issue https://github.com/pytorch/vision/issues/1712
RUN pip install torch
RUN pip install torchvision
RUN pip install torchaudio
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
RUN pip install tqdm
RUN pip install pytorch-lightning
RUN pip install yacs
RUN pip install --upgrade torch-geometric


RUN echo 'here'