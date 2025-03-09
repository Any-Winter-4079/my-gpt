FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up the environment
RUN apt update && apt install -y python3 python3-pip git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt

# Copy the project files
WORKDIR /app
COPY . /app

# Define entrypoint for training
CMD ["python3", "train_model.py"]