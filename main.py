"""  
Main script for the Virtual Dressing Room application.

@authors: 
    - Vanessa Nataly Manzano Estrada
    - Juan Daniel Alonzo Lopez
    - Joshua Nathaniel Arrazola Elizondo
"""

import cv2
import numpy as np
from src.config import SHIRT_DIRECTORY, WIDTH_SCALE, SHIRT_RATIO, SELECTION_THRESHOLD, CLOSET_PANEL_WIDTH
from src.overlay_utils import apply_overlay
from src.mediapipe_utils import init_mediapipe_modules
from src.closet_panel import load_closet_items, draw_closet_panel

mp_pose, mp_hands, mp_draw, pose_detector, hand_detector = init_mediapipe_modules()

cap = cv2.VideoCapture(0)

closet_items = load_closet_items(SHIRT_DIRECTORY)
if not closet_items:
    print("No se encontraron playeras en la carpeta del vestidor.")
    exit()

print("Playeras cargadas:")
for item in closet_items:
    print(item['filename'])

current_shirt_index = 0

selection_counter = 0
hovered_thumbnail_index = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame vacío, se omite...")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose_detector.process(rgb_frame)
    hand_results = hand_detector.process(rgb_frame)

    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        frame_h, frame_w, _ = frame.shape
        left_px = (int(left_shoulder.x * frame_w), int(left_shoulder.y * frame_h))
        right_px = (int(right_shoulder.x * frame_w), int(right_shoulder.y * frame_h))

        shirt_w = int(abs(left_px[0] - right_px[0]) * WIDTH_SCALE)
        shirt_h = int(shirt_w * SHIRT_RATIO)
        top_left_x = max(0, min(frame_w - shirt_w, min(left_px[0], right_px[0]) - int(shirt_w * 0.15)))
        top_left_y = max(0, min(frame_h - shirt_h, min(left_px[1], right_px[1]) - int(shirt_h * 0.2)))

        shirt_image = closet_items[current_shirt_index]['image']
        if shirt_image is not None:
            shirt_resized = cv2.resize(shirt_image, (shirt_w, shirt_h))
            frame = apply_overlay(frame, shirt_resized, top_left_x, top_left_y)

        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame, thumb_boxes = draw_closet_panel(frame, closet_items, current_shirt_index)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x = int(index_tip.x * frame.shape[1])
            finger_y = int(index_tip.y * frame.shape[0])
            
            if finger_x < CLOSET_PANEL_WIDTH:
                for idx, (x1, y1, x2, y2) in enumerate(thumb_boxes):
                    if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                        if hovered_thumbnail_index == idx:
                            selection_counter += 1
                        else:
                            hovered_thumbnail_index = idx
                            selection_counter = 1
                        
                        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 15, (0, 0, 255), 3)
                        if selection_counter >= SELECTION_THRESHOLD:
                            current_shirt_index = idx
                            selection_counter = 0
                        break
                else:
                    hovered_thumbnail_index = None
                    selection_counter = 0
            else:
                hovered_thumbnail_index = None
                selection_counter = 0

    cv2.imshow("Virtual Dressing Room", frame)
    if cv2.waitKey(5) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
