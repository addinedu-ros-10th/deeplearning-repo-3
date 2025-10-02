from gtts import gTTS
import uuid

async def speak(text: str) -> str:
    filename = "answer.mp3"
    print("TTS 시작")
    tts = gTTS(text=text, lang="ko")
    tts.save(filename)
    print("TTS 파일 저장완료", filename)
    return filename
