# Base image: Use Python 3.9 with CUDA if you're using GPUs
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    unzip \
    python3 \
    python3-pip

RUN git clone --depth 1 --branch main https://github.com/pytorch/examples.git /app/pytorch-examples

COPY tiny-imagenet-200.zip /app/tiny-imagenet-200.zip

RUN unzip /app/tiny-imagenet-200.zip -d /app/


# Copy project files to the container


COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY run_training_with_metrics.py /app/
COPY graph_gpu_metrics.py /app/
COPY pytorch-examples/imagenet/main.py /app/pytorch-examples/imagenet/main.py

# Set the default command to run
CMD ["python3", "project1/run_training_with_metrics.py", "--data-path", "/app/tiny-imagen