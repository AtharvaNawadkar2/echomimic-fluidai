import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from google.cloud import texttospeech
import openai
import uvicorn
import gradio as gr
from webgui1 import demo, process_video

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the Gradio app
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/google_cloud_credentials.json"

# Set up OpenAI API key
openai.api_key = "your_openai_api_key"

class TextToVideoRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>EchoMimic API</title>
        </head>
        <body>
            <h1>Welcome to EchoMimic API</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/gradio">Gradio Interface</a></li>
                <li>POST /audio_to_video</li>
                <li>POST /text_to_video</li>
            </ul>
        </body>
    </html>
    """

@app.post("/audio_to_video")
async def audio_to_video(audio: UploadFile = File(...), image: UploadFile = File(...)):
    # Save uploaded files temporarily
    audio_path = f"/tmp/{audio.filename}"
    image_path = f"/tmp/{image.filename}"
    
    with open(audio_path, "wb") as audio_file:
        audio_file.write(await audio.read())
    with open(image_path, "wb") as image_file:
        image_file.write(await image.read())

    try:
        # Process video using EchoMimic
        output_path = process_video(
            image_path, audio_path,
            width=512, height=512, length=1200, seed=420,
            facemask_dilation_ratio=0.1, facecrop_dilation_ratio=0.5,
            context_frames=12, context_overlap=3, cfg=2.5,
            steps=30, sample_rate=16000, fps=24, device="cuda"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        os.remove(audio_path)
        os.remove(image_path)

    return FileResponse(output_path, media_type="video/mp4", filename="output.mp4")

@app.post("/text_to_video")
async def text_to_video(request: TextToVideoRequest, image: UploadFile = File(...)):
    # Generate SSML using OpenAI's GPT-3.5 model
    ssml_prompt = f"Generate SSML for the following text to make it sound more lifelike. Include appropriate pauses, emphasis, and intonation: {request.text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in generating SSML (Speech Synthesis Markup Language) to make text sound more natural and lifelike when spoken by a text-to-speech system."},
            {"role": "user", "content": ssml_prompt}
        ]
    )
    ssml_text = response.choices[0].message.content.strip()

    # Generate audio using Google TTS
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the audio file temporarily
    audio_path = "/tmp/generated_audio.mp3"
    with open(audio_path, "wb") as out:
        out.write(response.audio_content)

    # Save the uploaded image temporarily
    image_path = f"/tmp/{image.filename}"
    with open(image_path, "wb") as image_file:
        image_file.write(await image.read())

    try:
        # Process video using EchoMimic
        output_path = process_video(
            image_path, audio_path,
            width=512, height=512, length=1200, seed=420,
            facemask_dilation_ratio=0.1, facecrop_dilation_ratio=0.5,
            context_frames=12, context_overlap=3, cfg=2.5,
            steps=30, sample_rate=16000, fps=24, device="cuda"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        os.remove(audio_path)
        os.remove(image_path)

    return FileResponse(output_path, media_type="video/mp4", filename="output.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
