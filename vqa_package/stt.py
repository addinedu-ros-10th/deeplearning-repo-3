import soundfile as sf
import numpy as np
import whisper
import sounddevice as sd   # PyAudio 대신 sounddevice 사용 권장

RATE = 16000
CHANNELS = 1
DURATION = 5   # 녹음 시간 (초)
OUTPUT_FILE = "temp.wav"

# Whisper 모델 미리 로드
stt_model = whisper.load_model("base")

def record_once():
    print(f"🎤 {DURATION}초 동안 말하세요...")
    recording = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=CHANNELS, dtype='int16')
    sd.wait()  # 녹음 끝날 때까지 대기
    print("🛑 녹음 종료")

    audio_np = recording.astype(np.float32) / 32768.0
    sf.write(OUTPUT_FILE, audio_np, RATE)

    result = stt_model.transcribe(OUTPUT_FILE, language="ko")
    return result["text"]
