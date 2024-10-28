import os
import io
import numpy as np
import librosa

import uvicorn
from fastapi import FastAPI, Form, File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ShanghaineseASR import ShanghaineseASR
from app.asr import ASR

class Message(BaseModel):
    input: str
    output: str = None

app = FastAPI()
asr = None
#asr = ASR()

audio_recognition_model_path = "E:/shanghainese_model/whisper-small-shanghainese"
shanghaineseASR = ShanghaineseASR(audio_recognition_model_path=audio_recognition_model_path)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/transcribe/")
async def  transcribe(message: Message):
    wav_file_path = message.input
    if wav_file_path is None or len(wav_file_path) <= 0:
        return {"code": 500, "msg": "audio file path MUST NOT be null or empty.", "output": ""}
    if not file_exists(wav_file_path):
        return {"code": 500, "msg": "audio file doesn't exists, please check it out.", "output": ""}
    try:
        transcribe_json_data = asr.transcribe(wav=wav_file_path)
        transcription = transcribe_json_data['transcription']
        translation = asr.translation(text=transcription)
    except:
        return {"code": 500, "msg": "server error.", "output": ""}
    transcribe_json_data['possible_translation'] = translation
    message.output = transcribe_json_data
    return {"code": 200, "msg": "success", "output": message.output}


@app.post("/transcribe2/")
async def transcribe2(audio_file: UploadFile = File(...)):
    if audio_file is None:
        return {"code": 500, "msg": "Please upload a audio file."}
    audio_file_bytes = await audio_file.read()
    translation = shanghaineseASR.transcribe_bytes(audio_file_bytes=audio_file_bytes)
    if translation is None or len(translation) <= 0:
        return {"code": 500, "msg": "failed", "translation": ""}
    return {"code": 200, "msg": "success", "translation": translation}

def file_exists(file_path):
    return os.path.exists(file_path)

if __name__ == '__main__':
  uvicorn.run('app.main:app', port=9999)