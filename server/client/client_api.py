import cv2
import requests
import os
import threading

import sounddevice as sd
import wave
import numpy as np


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
    recorded = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recorded.append(indata.copy())

    with sd.InputStream(samplerate=rate, channels=1, callback=callback, dtype='int16'):
        sd.sleep(duration * 1000)

    audio = np.concatenate(recorded, axis=0)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())

    print("녹음 완료")
    return filename


def send_vqa(frame_path):
    def worker():
        audio_file = record_audio()
        with open(audio_file, "rb") as a, open(frame_path, "rb") as img:
            response = requests.post(f"{SERVER_URL}/vqa/", files={"audio": a, "image": img})
        with open("answer.mp3", "wb") as f:
            f.write(response.content)
        print("음성 재생")
        os.system("mpg123 answer.mp3")

    # 백그라운드 스레드 실행
    threading.Thread(target=worker, daemon=True).start()