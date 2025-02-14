
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the model architecture from the JSON file
try:
    with open("newemotiondetector.json", "r") as json_file:
        model_json = json_file.read()
        if not model_json.strip():
            raise ValueError("The JSON file is empty.")
except FileNotFoundError:
    print("The JSON file was not found.")
    exit()
except ValueError as ve:
    print(f"ValueError: {ve}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the JSON file: {e}")
    exit()

# Create the model from the JSON architecture
try:
    model = model_from_json(model_json)
except json.JSONDecodeError as jde:
    print(f"JSONDecodeError: {jde}")
    exit()
except Exception as e:
    print(f"An error occurred while creating the model: {e}")
    exit()

# Load the model weights
try:
    model.load_weights("newemotiondetector.h5")
except FileNotFoundError:
    print("The weights file was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the weights: {e}")
    exit()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize the webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            print("Predicted Output:", prediction_label)
            cv2.putText(im, prediction_label, (p, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Output", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        pass

webcam.release()
cv2.destroyAllWindows()