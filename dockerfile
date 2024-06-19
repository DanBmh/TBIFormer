FROM nvcr.io/nvidia/pytorch:22.11-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends feh
RUN apt-get update && apt-get install -y --no-install-recommends python3-opencv
RUN pip uninstall -y opencv-python && pip install --no-cache "opencv-python<4.3"

RUN apt-get update && apt-get install -y --no-install-recommends python3-tk
RUN pip install --upgrade --no-cache-dir pip

# Install TBIFormer
RUN git clone https://github.com/xiaogangpeng/tbiformer.git
RUN pip install --no-cache open3d
RUN pip install --upgrade --no-cache werkzeug flask
RUN pip install --no-cache easydict
RUN pip install --no-cache torch-dct
RUN echo "ldconfig" >> ~/.bashrc

WORKDIR /tbiformer/
CMD ["/bin/bash"]
