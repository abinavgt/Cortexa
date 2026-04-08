import cv2
import numpy as np
import os
import time
import winsound
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the numerical to emotion string mapping based on the notebook
# emotion_mapping = {"fear": 0, "neutral": 1, "sad": 2, "angry": 3, "happy": 4, "disgust": 5, "surprise": 6}
model = tf.keras.models.load_model('cnn_model.h5')
emotion_dict = {
    0: "Fear",
    1: "Neutral",
    2: "Sad",
    3: "Angry",
    4: "Happy",
    5: "Disgust",
    6: "Surprise"
}

# 1. Load the pre-trained model. We assume it is saved as 'emotion_model.h5'
# The notebook actually saves 'emotion_model.h5'. Let the user know if missing.
model_path = 'emotion_model.h5'
if not os.path.exists(model_path):
    # Try alternative name that might be used
    if os.path.exists('cnn_model.h5'):
        model_path = 'cnn_model.h5'
    else:
        print(f"Error: Model file not found at {model_path}.")
        print("Please ensure your trained Keras model (e.g. emotion_model.h5) is saved in this directory.")
        exit(1)

print(f"Loading model from {model_path}...")
model = load_model(model_path)
print("Model loaded successfully.")

# 2. Load the Haar cascade for face and eye detection
# OpenCV comes with pre-trained haarcascades, we can access them via cv2.data.haarcascades
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(cascade_path)

eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye_tree_eyeglasses.xml')
eye_classifier = cv2.CascadeClassifier(eye_cascade_path)

# Variables for eye closure detection
closed_start_time = None
last_alert_time = 0

# Initialize video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

print("Starting video feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
        
    # Convert frame to grayscale as the model expects 1-channel image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Reset timer if no face is detected
    if len(faces) == 0:
        closed_start_time = None

    for (x, y, w, h) in faces:
        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the Region of Interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        
        # To reduce false positives (like closed eyes being detected as open),
        # only search the upper 60% of the face, and increase minNeighbors and minSize.
        roi_gray_eyes = gray[y:y+int(h*0.6), x:x+w]
        
        # Detect eyes in the upper face ROI
        eyes = eye_classifier.detectMultiScale(roi_gray_eyes, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
        
        # If no eyes are detected, we assume they are closed
        if len(eyes) == 0:
            if closed_start_time is None:
                closed_start_time = time.time()
            elif (time.time() - closed_start_time) >= 3.0:
                cv2.putText(frame, "DROWSINESS DETECTED!", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if (time.time() - last_alert_time) > 1.0:  # Play beep every 1 second
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                    last_alert_time = time.time()
        else:
            closed_start_time = None
            # Draw rectangles around the eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
        
        # Resize to 48x48 (expected input size for the CNN)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Normalize pixel values
        roi_gray = roi_gray.astype('float32') / 255.0
        
        # Expand dimensions to match the input shape of the model: (batch_size, height, width, channels)
        # So we make it (1, 48, 48, 1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        # Predict the emotion
        prediction = model.predict(roi_gray, verbose=0)
        maxindex = int(np.argmax(prediction))
        predicted_emotion = emotion_dict[maxindex]
        
        # Display the predicted emotion text on the video frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Show the video feed with the bounding box and emotion label
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
