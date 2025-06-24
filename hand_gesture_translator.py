import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr
from deep_translator import GoogleTranslator  # Updated

# Initialize MediaPipe for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize Speech-to-Text recognizer
recognizer = sr.Recognizer()

# Function to translate text
def translate_text(text, target_language='es'):
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        print(f"Translated Text: {translated}")
        return translated
    except Exception as e:
        print("Translation error:", e)
        return text

# Convert Gesture to Text and Speech
def gesture_to_speech_with_translation():
    print("Choose a target language (e.g., 'es' for Spanish, 'fr' for French, 'de' for German):")
    target_language = input("Enter language code: ")

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        className = ""

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Hand Gesture Recognition with Translation", frame)

        if className:
            print(f"Recognized gesture: {className}")
            translated_text = translate_text(className, target_language=target_language)
            engine.say(translated_text)
            engine.runAndWait()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Convert Gesture to Text
def gesture_to_text():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        className = ""

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Gesture Recognition", frame)

        if className:
            print(f"Recognized gesture: {className}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Convert Speech to Sign Language
def speech_to_sign_language():
    with sr.Microphone() as source:
        print("Please speak something...")
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")
            display_sign_language(text)
        except Exception as e:
            print("Sorry, I couldn't recognize that. Try again!")

def display_sign_language(text):
    print(f"Displaying sign language for: {text}")
    # Placeholder: Could be replaced with sign animation or visual

# Convert Text to Speech
def text_to_speech():
    text = input("Enter text to convert to speech: ")
    engine.say(text)
    engine.runAndWait()

# Main Program
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Convert Hand Gesture to Speech with Translation")
    print("2. Convert Hand Gesture to Text (No Speech)")
    print("3. Convert Speech to Sign Language")
    print("4. Convert Text to Speech")

    choice = input("Enter your choice (1/2/3/4): ")

    if choice == "1":
        gesture_to_speech_with_translation()
    elif choice == "2":
        gesture_to_text()
    elif choice == "3":
        speech_to_sign_language()
    elif choice == "4":
        text_to_speech()
    else:
        print("Invalid choice!")
