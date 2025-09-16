from gtts import gTTS
import playsound
import os

def tts(text):
    tts = gTTS(text=text, lang="ko")
    tts.save("output.mp3")

    print("음성 출력 중...")
    playsound.playsound("output.mp3")
    print("음성 출력 종료")
    # os.remove(output_mp3)