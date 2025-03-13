"""  
Module for overlaying an image on top of another image.

This module provides a function to overlay an image on top of another image.
The overlay image can have an alpha channel to control transparency.
"""

import cv2
import numpy as np

"""  
Function to apply an overlay image on top of a background image at a specified position.

Parameters:
- background: the background image to overlay on
- overlay: the overlay image to apply
- pos_x: the x-coordinate position to place the overlay
- pos_y: the y-coordinate position to place the overlay

Returns:
- result: the resulting image with the overlay applied
"""
def apply_overlay(background, overlay, pos_x, pos_y):
    bg_h, bg_w = background.shape[:2]

    pos_x = max(0, min(pos_x, bg_w - 1))
    pos_y = max(0, min(pos_y, bg_h - 1))

    overlay_h, overlay_w = overlay.shape[:2]
    if pos_x + overlay_w > bg_w:
        overlay_w = bg_w - pos_x
    if pos_y + overlay_h > bg_h:
        overlay_h = bg_h - pos_y

    if overlay_w <= 0 or overlay_h <= 0:
        return background

    resized_overlay = cv2.resize(overlay, (overlay_w, overlay_h))
    if resized_overlay.shape[2] < 4:
        alpha_channel = 255 * np.ones((resized_overlay.shape[0], resized_overlay.shape[1], 1), dtype=resized_overlay.dtype)
        resized_overlay = np.concatenate([resized_overlay, alpha_channel], axis=2)

    overlay_rgb = resized_overlay[..., :3]
    alpha_mask = resized_overlay[..., 3:] / 255.0

    background[pos_y:pos_y+overlay_h, pos_x:pos_x+overlay_w] = \
        (1 - alpha_mask) * background[pos_y:pos_y+overlay_h, pos_x:pos_x+overlay_w] + alpha_mask * overlay_rgb

    return background
