# Use the base image
FROM samge/echo_mimic-dev-base

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install additional dependencies
RUN pip install fastapi uvicorn google-cloud-texttospeech

# Set up environment variables
ENV PYTHONPATH=/app
ENV FFMPEG_PATH=/home/ffmpeg-release-amd64-static

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    