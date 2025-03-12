import cv2
import numpy as np

def apply_overlay(bg_image, overlay_img, pos_x, pos_y):
    bg_height, bg_width = bg_image.shape[:2]

    pos_x = max(0, min(pos_x, bg_width - 1))
    pos_y = max(0, min(pos_y, bg_height - 1))

    overlay_height, overlay_width = overlay_img.shape[:2]
    if pos_x + overlay_width > bg_width:
        overlay_width = bg_width - pos_x
    if pos_y + overlay_height > bg_height:
        overlay_height = bg_height - pos_y

    if overlay_width <= 0 or overlay_height <= 0:
        return bg_image

    resized_overlay = cv2.resize(overlay_img, (overlay_width, overlay_height))
    if resized_overlay.shape[2] < 4:
        alpha_layer = np.ones((resized_overlay.shape[0], resized_overlay.shape[1], 1), dtype=resized_overlay.dtype) * 255
        resized_overlay = np.concatenate([resized_overlay, alpha_layer], axis=2)

    overlay_rgb = resized_overlay[..., :3]
    alpha_mask = resized_overlay[..., 3:] / 255.0

    bg_image[pos_y:pos_y+overlay_height, pos_x:pos_x+overlay_width] = \
        (1 - alpha_mask) * bg_image[pos_y:pos_y+overlay_height, pos_x:pos_x+overlay_width] + alpha_mask * overlay_rgb

    return bg_image
