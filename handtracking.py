import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open video feed or load an image
cap = cv2.VideoCapture(0)  # For webcam input, use 0. For video, replace with file path.

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert the image color space from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the output
    cv2.imshow("Hand Tracking", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
