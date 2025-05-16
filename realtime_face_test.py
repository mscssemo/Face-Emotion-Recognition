

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('models/model1.h5')

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Categories order (copy this from your training script’s output)
categories = ['Angry', 'Disgust', 'Neutral', 'Sad', 'Smiling', 'Surprise']

# Define target size (same as model input)
target_size = (64, 64)

# Start webcam
cap = cv2.VideoCapture(0)

start_time = time.time()
duration = 1 * 60  # 1 minutes in seconds

while True:

    if time.time() - start_time > duration:
        print("1 minutes have passed. Exiting...")
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face ROI in RGB
        roi_bgr_frame = frame[y:y + h, x:x + w]
        roi_rgb_frame = cv2.cvtColor(roi_bgr_frame, cv2.COLOR_BGR2RGB)

        # Resize face to match model input size
        resized_face = cv2.resize(roi_rgb_frame, target_size)
        normalized_face = resized_face / 255.0
        face_array = np.expand_dims(normalized_face, axis=0)  # (1, 128, 128, 3)

        # Predict emotion
        prediction = model.predict(face_array)
        predicted_class = np.argmax(prediction)
        label_text = categories[predicted_class]

        # Display prediction label
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255, 0, 0), 2)

    # Show the live frame
    cv2.imshow('Live Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy windows
cap.release()
cv2.destroyAllWindows()
