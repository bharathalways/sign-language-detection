import pickle
import cv2
import mediapipe as mp
from collections import deque

# Load the trained model
with open('sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam
cap = cv2.VideoCapture(0)

# Settings for letter stabilization
buffer = deque(maxlen=15)
current_letter = ""
word = ""
last_prediction = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                buffer.append(prediction)

                # If the majority of recent predictions are the same, accept it
                if buffer.count(prediction) > 10:
                    if prediction != last_prediction:
                        word += prediction
                        last_prediction = prediction
                        buffer.clear()

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display the word being formed
    cv2.putText(frame, f'Word: {word}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow('Sign Language to Word', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 8:  # Backspace
        word = word[:-1]
    elif key == 32:  # Spacebar
        word += " "

cap.release()
cv2.destroyAllWindows()
