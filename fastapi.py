# app/main.py
import os
import tempfile
from pathlib import Path
from typing import Optional
from EchoMimic.fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from google.cloud import texttospeech
from openai import OpenAI
import subprocess
import json
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid

app = FastAPI()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Google Cloud TTS client
tts_client = texttospeech.TextToSpeechClient()

# Initialize EchoMimic pipeline
# This would be similar to what's in the webgui.py but without the Gradio interface
pipe = initialize_echomimic_pipeline()  # You'll need to implement this based on webgui.py

def generate_ssml(text: str) -> str:
    """Generate SSML markup using OpenAI"""
    prompt = f"""
    Convert the following text into SSML markup to make it sound more natural and expressive.
    Include appropriate breaks, prosody, and emphasis tags.
    Text: {text}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def text_to_speech(ssml: str) -> bytes:
    """Convert SSML to audio using Google Cloud TTS"""
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-F",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    return response.audio_content

@app.post("/process/")
async def process_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """Process video using EchoMimic with provided image and audio"""
    try:
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            image_path = Path(temp_dir) / "input_image.jpg"
            audio_path = Path(temp_dir) / "input_audio.mp3"
            
            with open(image_path, "wb") as f:
                f.write(await image.read())
            with open(audio_path, "wb") as f:
                f.write(await audio.read())
            
            # Process using EchoMimic
            output_path = Path(temp_dir) / "output_video.mp4"
            video = pipe(
                image_path,
                audio_path,
                output_path
            )
            
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename="output_video.mp4"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/")
async def generate_video(
    image: UploadFile = File(...),
    text: str = Form(...),
):
    """Generate video from image and text using TTS and EchoMimic"""
    try:
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            image_path = Path(temp_dir) / "input_image.jpg"
            with open(image_path, "wb") as f:
                f.write(await image.read())
            
            # Generate SSML from text
            ssml = generate_ssml(text)
            
            # Convert SSML to audio
            audio_content = text_to_speech(ssml)
            
            # Save audio file
            audio_path = Path(temp_dir) / "generated_audio.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_content)
            
            # Process using EchoMimic
            output_path = Path(temp_dir) / "output_video.mp4"
            video = pipe(
                image_path,
                audio_path,
                output_path
            )
            
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename="output_video.mp4"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}