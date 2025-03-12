import cv2
import os
import numpy as np

from src.config import SHIRT_DIRECTORY, BUTTON_IMAGE_PATH, WIDTH_SCALE, SHIRT_RATIO, NAV_SPEED
from src.overlay_utils import apply_overlay
from src.mediapipe_utils import init_mediapipe_modules

# Inicialización de MediaPipe
mp_pose, mp_hands, mp_draw, pose_detector, hand_detector = init_mediapipe_modules()

# Inicializar cámara
video_stream = cv2.VideoCapture(0)

# Cargar imágenes de camisas y botones
shirt_list = os.listdir(SHIRT_DIRECTORY)
print("Available shirts:", shirt_list)
print(f"Total shirts found: {len(shirt_list)}")

current_shirt = 0

button_img_right = cv2.imread(BUTTON_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
if button_img_right is None:
    print("Error loading button image. Check the path.")
    exit()
button_img_left = cv2.flip(button_img_right, 1)

nav_counter_right = 0
nav_counter_left = 0

while video_stream.isOpened():
    ret, frame = video_stream.read()
    if not ret:
        print("Empty frame, skipping...")
        continue

    # Espejar la imagen para efecto selfie
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar imagen con MediaPipe
    pose_results = pose_detector.process(rgb_frame)
    hand_results = hand_detector.process(rgb_frame)
    
    if pose_results.pose_landmarks:
        left_shldr = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shldr = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        frame_height, frame_width, _ = frame.shape
        left_px = (int(left_shldr.x * frame_width), int(left_shldr.y * frame_height))
        right_px = (int(right_shldr.x * frame_width), int(right_shldr.y * frame_height))
        
        # Calcular dimensiones y posición de la camisa
        calculated_width = int(abs(left_px[0] - right_px[0]) * WIDTH_SCALE)
        calculated_height = int(calculated_width * SHIRT_RATIO)
        top_left_x = max(0, min(frame_width - calculated_width, min(left_px[0], right_px[0]) - int(calculated_width * 0.15)))
        top_left_y = max(0, min(frame_height - calculated_height, min(left_px[1], right_px[1]) - int(calculated_height * 0.2)))
        
        shirt_path = os.path.join(SHIRT_DIRECTORY, shirt_list[current_shirt])
        shirt_image = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        if shirt_image is not None:
            shirt_image = cv2.resize(shirt_image, (calculated_width, calculated_height))
            frame = apply_overlay(frame, shirt_image, top_left_x, top_left_y)
        
        # Dibujo de landmarks para depuración
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Superponer botones de navegación
    right_button_pos = (frame.shape[1] - button_img_right.shape[1] - 10, 
                        frame.shape[0] // 2 - button_img_right.shape[0] // 2)
    left_button_pos = (10, frame.shape[0] // 2 - button_img_left.shape[0] // 2)
    frame = apply_overlay(frame, button_img_right, *right_button_pos)
    frame = apply_overlay(frame, button_img_left, *left_button_pos)
    
    # Detección de interacción con botones usando landmarks de manos
    if hand_results.multi_hand_landmarks:
        for hand_landmark in hand_results.multi_hand_landmarks:
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_x = int(index_tip.x * frame.shape[1])
            tip_y = int(index_tip.y * frame.shape[0])
            
            # Botón derecho
            if tip_x > frame.shape[1] - button_img_right.shape[1] - 10 and \
               frame.shape[0] // 2 - button_img_right.shape[0] // 2 < tip_y < frame.shape[0] // 2 + button_img_right.shape[0] // 2:
                nav_counter_right += 1
                cv2.ellipse(frame, (frame.shape[1] - button_img_right.shape[1] // 2 - 10, frame.shape[0] // 2),
                            (66, 66), 0, 0, nav_counter_right * NAV_SPEED, (0, 255, 0), 20)
                if nav_counter_right * NAV_SPEED > 360:
                    nav_counter_right = 0
                    current_shirt = (current_shirt + 1) % len(shirt_list)
                    print(f"Switched to next shirt: {current_shirt}")
            # Botón izquierdo
            elif tip_x < button_img_left.shape[1] + 10 and \
                 frame.shape[0] // 2 - button_img_left.shape[0] // 2 < tip_y < frame.shape[0] // 2 + button_img_left.shape[0] // 2:
                nav_counter_left += 1
                cv2.ellipse(frame, (button_img_left.shape[1] // 2 + 10, frame.shape[0] // 2),
                            (66, 66), 0, 0, nav_counter_left * NAV_SPEED, (0, 255, 0), 20)
                if nav_counter_left * NAV_SPEED > 360:
                    nav_counter_left = 0
                    current_shirt = current_shirt - 1 if current_shirt > 0 else len(shirt_list) - 1
                    print(f"Switched to previous shirt: {current_shirt}")
            else:
                nav_counter_right = 0
                nav_counter_left = 0
                
    cv2.imshow('Virtual Try-On', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

video_stream.release()
cv2.destroyAllWindows()
