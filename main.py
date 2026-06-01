import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Webcam setup
cap = cv2.VideoCapture(0)

# Lower resolution for better speed
cap.set(3, 640)
cap.set(4, 480)

# Screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

draw = mp.solutions.drawing_utils

# Smooth cursor movement
prev_x = 0
prev_y = 0
smoothening = 8

# Click cooldown
last_click_time = 0
click_delay = 1

# Drag state
dragging = False

# FPS variables
prev_time = 0
curr_time = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    result = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks
            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Fingertips
            thumb = landmarks[4]
            index = landmarks[8]
            middle = landmarks[12]

            # Finger joints
            index_joint = landmarks[6]
            middle_joint = landmarks[10]

            # Coordinates
            tx = int(thumb.x * frame_width)
            ty = int(thumb.y * frame_height)

            ix = int(index.x * frame_width)
            iy = int(index.y * frame_height)

            mx = int(middle.x * frame_width)
            my = int(middle.y * frame_height)

            # Draw circles
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 10, (255, 0, 0), -1)
            cv2.circle(frame, (mx, my), 10, (0, 0, 255), -1)

            # Convert webcam coordinates to screen coordinates
            screen_x = np.interp(ix, [0, frame_width], [0, screen_width])
            screen_y = np.interp(iy, [0, frame_height], [0, screen_height])

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            # Move mouse
            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            # Distances
            thumb_index_distance = math.hypot(tx - ix, ty - iy)
            thumb_middle_distance = math.hypot(tx - mx, ty - my)

            current_time = time.time()

            # ---------------------------------
            # LEFT CLICK
            # ---------------------------------
            if thumb_index_distance < 20 and not dragging:

                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                if current_time - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = current_time

            # ---------------------------------
            # RIGHT CLICK
            # ---------------------------------
            elif thumb_middle_distance < 20:

                cv2.putText(
                    frame,
                    "RIGHT CLICK",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                if current_time - last_click_time > click_delay:
                    pyautogui.rightClick()
                    last_click_time = current_time

            # ---------------------------------
            # DRAG MODE
            # ---------------------------------
            drag_distance = math.hypot(ix - mx, iy - my)

            if (
                thumb_index_distance < 35 and
                thumb_middle_distance < 35 and
                drag_distance < 35
            ):

                cv2.putText(
                    frame,
                    "DRAG MODE",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True

            else:

                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # ---------------------------------
            # SCROLL MODE
            # ---------------------------------
            index_up = index.y < index_joint.y
            middle_up = middle.y < middle_joint.y

            if index_up and middle_up and thumb_index_distance > 50:

                cv2.putText(
                    frame,
                    "SCROLL MODE",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                # Scroll Up
                if iy < frame_height // 2:
                    pyautogui.scroll(20)

                # Scroll Down
                else:
                    pyautogui.scroll(-20)

    # ---------------------------------
    # FPS Calculation
    # ---------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # FPS Display
    cv2.putText(
        frame,
        f'FPS: {int(fps)}',
        (500, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Project Title
    cv2.putText(
        frame,
        "AI Virtual Mouse",
        (20, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Webcam window
    cv2.imshow("AI Virtual Mouse", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()