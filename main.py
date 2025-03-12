"""
Main module for the Virtual Dressing Room application.

Authors: 
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

def overlay_shirt_on_frame(frame, pose_result, closet_items, selected_index, mp_pose, mp_draw):
    """Superpone la camisa seleccionada en función de la detección de pose."""
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        h, w, _ = frame.shape
        left_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        shirt_width = int(abs(left_coords[0] - right_coords[0]) * WIDTH_SCALE)
        shirt_height = int(shirt_width * SHIRT_RATIO)
        x_offset = max(0, min(w - shirt_width, min(left_coords[0], right_coords[0]) - int(shirt_width * 0.15)))
        y_offset = max(0, min(h - shirt_height, min(left_coords[1], right_coords[1]) - int(shirt_height * 0.2)))
        
        shirt_img = closet_items[selected_index]['image']
        if shirt_img is not None:
            resized_shirt = cv2.resize(shirt_img, (shirt_width, shirt_height))
            frame = apply_overlay(frame, resized_shirt, x_offset, y_offset)
        mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame

def process_hand_selection(frame, hand_result, panel_boxes, current_index, dwell_counter, mp_hands):
    """
    Procesa la selección de la camisa mediante la interacción con la mano.
    Si el dedo índice se posiciona sobre una miniatura por cierto número de frames,
    se actualiza la camisa seleccionada.
    """
    for hand_landmarks in hand_result.multi_hand_landmarks:
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x_tip = int(tip.x * frame.shape[1])
        y_tip = int(tip.y * frame.shape[0])
        if x_tip < CLOSET_PANEL_WIDTH:
            for idx, (x1, y1, x2, y2) in enumerate(panel_boxes):
                if x1 <= x_tip <= x2 and y1 <= y_tip <= y2:
                    if current_index == idx:
                        dwell_counter = 0  
                    else:
                        dwell_counter += 1
                        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 15, (0, 0, 255), 3)
                        if dwell_counter >= SELECTION_THRESHOLD:
                            current_index = idx
                            dwell_counter = 0
                    break
            else:
                dwell_counter = 0
        else:
            dwell_counter = 0
    return frame, current_index, dwell_counter

def main():
    mp_pose, mp_hands, mp_draw, pose_detector, hand_detector = init_mediapipe_modules()
    cap = cv2.VideoCapture(0)
    closet_items = load_closet_items(SHIRT_DIRECTORY)
    if not closet_items:
        print("No se encontraron playeras en el vestidor.")
        return

    selected_shirt = 0
    dwell_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_out = pose_detector.process(rgb_frame)
        hand_out = hand_detector.process(rgb_frame)
        
        frame = overlay_shirt_on_frame(frame, pose_out, closet_items, selected_shirt, mp_pose, mp_draw)
        frame, thumb_boxes = draw_closet_panel(frame, closet_items, selected_shirt)
        
        if hand_out.multi_hand_landmarks:
            frame, selected_shirt, dwell_counter = process_hand_selection(frame, hand_out, thumb_boxes, selected_shirt, dwell_counter, mp_hands)
        
        cv2.imshow("Virtual Dressing Room", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
