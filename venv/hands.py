import cv2
import mediapipe as mp
# import numpy as np
# import tensorflow as tf

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_drawing_utils = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing_utils.draw_landmarks(
                img,hand_landmark, mp_hands.HAND_CONNECTIONS
                )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()