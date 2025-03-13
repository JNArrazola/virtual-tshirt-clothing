"""  
Module to initialize mediapipe modules and objects

This module provides a function to initialize mediapipe modules and objects for pose and hand detection.
"""

import mediapipe as mp

"""  
Create and initialize mediapipe modules and objects
"""
def init_mediapipe_modules():
    pose_module = mp.solutions.pose
    hands_module = mp.solutions.hands
    drawing_module = mp.solutions.drawing_utils

    # Initialize pose and hand detectors
    pose_detector = pose_module.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    # Initialize hand detector
    hand_detector = hands_module.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    return pose_module, hands_module, drawing_module, pose_detector, hand_detector
