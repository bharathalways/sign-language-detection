import cv2
import mediapipe as mp
import pickle
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create or load existing data
data = []
labels = []

if os.path.exists("data.pickle"):
    with open("data.pickle", "rb") as f:
        existing_data = pickle.load(f)
        data = existing_data["data"]
        labels = existing_data["labels"]

# Ask user for label
label = input("Enter the label for the sign (e.g., A, B, C): ")

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                data.append(landmarks)
                labels.append(label)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(image, f'Collecting for: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection - Press 'q' to stop", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save to pickle
with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Data saved to data.pickle")
