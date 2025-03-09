FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install additional dependencies
RUN pip install numpy tqdm tiktoken datasets

# Copy project files
COPY . /workspace