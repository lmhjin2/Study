from gtts import gTTS
import os

# 텍스트 입력
text = ""

# gTTS를 사용하여 음성 파일 생성
tts = gTTS(text=text, lang='en')

tts.save("C:/Users/AIA/Desktop/temp/tts.mp3")