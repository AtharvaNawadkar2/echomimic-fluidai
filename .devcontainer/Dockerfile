# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg
RUN wget https://johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz \
    && tar xf ffmpeg-4.4-amd64-static.tar.xz \
    && mv ffmpeg-4.4-amd64-static/ffmpeg /usr/local/bin/ \
    && mv ffmpeg-4.4-amd64-static/ffprobe /usr/local/bin/ \
    && chmod +x /usr/local/bin/ffmpeg \
    && chmod +x /usr/local/bin/ffprobe \
    && rm -rf ffmpeg-4.4-amd64-static*

# Set environment variables
ENV FFMPEG_PATH=/usr/local/bin
ENV PATH="${FFMPEG_PATH}:${PATH}"

# Clone EchoMimic repository
RUN git clone https://github.com/BadToBest/EchoMimic .

# Install Python dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install fastapi uvicorn google-cloud-texttospeech python-multipart openai

# Download pre-trained models
RUN git lfs install
RUN git clone https://huggingface.co/BadToBest/EchoMimic pretrained_weights

# Copy application code
COPY ./app /app/src

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
