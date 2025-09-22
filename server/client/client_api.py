import cv2
import requests
import os

import sounddevice as sd
import wave


SERVER_URL = "http://192.168.0.132:8000"

def send_caption(frame_path):
    with open(frame_path, "rb") as img:
        response = requests.post(f"{SERVER_URL}/caption/", files={"image": img})
    with open("answer.mp3", "wb") as f:
        f.write(response.content)
    print("음성 재생")
    os.system("mpg123 answer.mp3")
    

def record_audio(filename="question.wav", duration=5, rate=16000):
    print("🎤 질문을 말씀하세요...")
    recording = sd.rec(int(duration*rate), samplerate=rate, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(recording.tobytes())
    print("녹음 완료")
    return filename

def send_vqa(frame_path):
    audio_file = record_audio()
    with open(audio_file, "rb") as a, open(frame_path, "rb") as img:
        response = requests.post(f"{SERVER_URL}/vqa/", files={"audio": a, "image": img})
    with open("answer.mp3", "wb") as f:
        f.write(response.content)
    print("음성 재생")
    os.system("mpg123 answer.mp3")
