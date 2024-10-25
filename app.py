import cv2
import numpy as np
import mediapipe as mp
import joblib
import time

# Load the pre-trained model
model = joblib.load('./sign_language.pkl')

# Class labels for A-Z and 0-9
class_names = {i: chr(65 + i) for i in range(26)}
class_names.update({i + 26: str(i) for i in range(10)})

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Time tracking for prediction intervals
last_prediction_time = time.time()
last_predicted_char = 'Waiting...'  # Initial display value

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Invert (mirror) the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB and process with MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand region
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            hand_frame = frame[y_min:y_max, x_min:x_max]

            # Make a prediction only if 1 second has passed
            if hand_frame.size != 0 and time.time() - last_prediction_time >= 1:
                try:
                    # Preprocess and predict
                    resized_frame = cv2.resize(hand_frame, (50, 50)) / 255.0
                    predictions = model.predict(np.expand_dims(resized_frame, axis=0))
                    predicted_class = np.argmax(predictions[0])
                    last_predicted_char = class_names.get(predicted_class, 'Unknown')

                    # Update the last prediction time
                    last_prediction_time = time.time()

                except:
                    last_predicted_char = 'Error'

    # Display the last predicted character continuously until the next prediction
    cv2.putText(frame, f'Prediction: {last_predicted_char}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Language', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
