import speech_recognition as sr
# Create a Recognizer instance
recognizer = sr.Recognizer()

def hear():
    # Capture audio input from the microphone
    with sr.Microphone() as source:
        print("Speak something...")
        audio_data = recognizer.listen(source)
    # Perform speech recognition using Google Web Speech API
    try:
        qn = recognizer.recognize_google(audio_data)
        print("You said:", qn)
    except:
        return hear()
    return qn


from gtts import gTTS
from playsound import playsound
import os
#Language in which you want to convert
language = 'en'

def speak(mytext):
    myobj = gTTS(text=mytext, lang=language, slow=False)
    # Saving the converted audio in a mp3 file named
    myobj.save("welcome.mp3")
    # Playing the converted file
    playsound("welcome.mp3")
    os.remove("welcome.mp3")
    return None

import google.generativeai as genai
GOOGLE_API_KEY = "AIzaSyCrUCzO2suUkeiwLR3NYXG85Erw1X4Zh00"
genai.configure(api_key=GOOGLE_API_KEY)
global gemini
gemini = genai.GenerativeModel('gemini-1.5-flash')

def to_markdown(text):
    text = text.replace('*','')
    return text

def solve(qn):
    response = gemini.generate_content(qn)
    if response.text:
        ans = to_markdown(response.text)
        return ans
    else:
        return False

import cv2
import face_recognition
import numpy as np
import pickle
## face-recognition
with open('encodings.pkl', 'rb') as f:
    all_encodings, classNames = pickle.load(f)
encodeListKnown = list(all_encodings.values())

def find_user():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if success:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            if not facesCurFrame:
                print("No faces detected")

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name, group_name = classNames[matchIndex]
                    cap.release()
                    return name

while True:
    name = find_user()
    speak(f"Hey {name}, ask me your question.")
    question = hear()
    if question.lower() in ['bye','exit']:
        break
    answer = solve(question)
    if answer:
        speak(f"Hear is your answer! {answer}")
    else:
        speak(f"Sorry {name}, I can't get you answer, please ask me different question.")
        continue
