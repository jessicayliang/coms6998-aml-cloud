# Base image: Use Python 3.9 with CUDA if you're using GPUs
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch main https://github.com/pytorch/examples.git /app/pytorch-examples


# Copy project files to the container


COPY requirements.txt /app/
RUN pip install enum34 --user

RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY run_training_with_metrics.py /app/
COPY graph_gpu_metrics.py /app/

# Set the default command to run
CMD ["python3", "/app/run_training_with_metrics.py", "--data-path", "/data/tiny-imagenet-200/"]
