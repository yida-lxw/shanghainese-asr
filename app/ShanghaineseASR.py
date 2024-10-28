# -*- coding: utf-8 -*-

import io
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, pipeline

class ShanghaineseASR:
    def __init__(self, audio_recognition_model_path:str, language:str = "Chinese", max_new_tokens:int=128, chunk_length:int=30):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        #audio_recognition_model_path = "E:/shanghainese_model/whisper-small-shanghainese"
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.chunk_length = chunk_length
        self.processor = AutoProcessor.from_pretrained(audio_recognition_model_path, language=language)
        self.pipe = pipeline(task="automatic-speech-recognition",
                        model=audio_recognition_model_path,
                        tokenizer=self.processor.tokenizer,
                        feature_extractor=self.processor.feature_extractor,
                        max_new_tokens=max_new_tokens,
                        chunk_length_s=chunk_length,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        device=device)

    def transcribe(self, audio_file:np.ndarray):
        result = self.pipe(audio_file, return_timestamps=True, generate_kwargs={
            "task": "transcribe",
            "language": self.language
        })
        if result is None or not 'text' in result:
            return None
        translation = result["text"]
        if translation is None or len(translation) <= 0:
            return None
        return translation

    def transcribe_bytes(self, audio_file_bytes:bytes):
        convert_result = self.convertToNdArray(audio_file_bytes)
        audio_data = convert_result[0]
        result = self.pipe(audio_data, return_timestamps=True, generate_kwargs={
            "task": "transcribe",
            "language": self.language
        })
        if result is None or not 'text' in result:
            return None
        translation = result["text"]
        if translation is None or len(translation) <= 0:
            return None
        return translation

    def convertToNdArray(self, audio_file_bytes:bytes, target_sample_rate:int=16000):
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_file_bytes), sr=None)
        if sample_rate != target_sample_rate:
            resampled_audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            return (resampled_audio_data, target_sample_rate,)
        return (audio_data, sample_rate,)

