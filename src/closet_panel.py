"""  
Module for loading and drawing the closet panel

This module provides functions to load closet items from a folder and draw the closet panel on the screen.
The closet panel is a vertical panel on the left side of the screen that shows the available closet items as thumbnails.
"""

import cv2
import os
import numpy as np
from src.config import CLOSET_PANEL_WIDTH, THUMBNAIL_SIZE

"""  
Function to load closet items from a folder

Parameters:
- folder_path: path to the folder containing the closet items

Returns:
- items: a list of dictionaries, each containing the following
"""
def load_closet_items(folder_path):
    items = []
    file_list = os.listdir(folder_path)
    for filename in file_list:
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            thumbnail = cv2.resize(img, THUMBNAIL_SIZE)
            items.append({
                'filename': filename,
                'image': img,
                'thumbnail': thumbnail
            })
    return items

"""  
Function to draw the closet panel on the screen

Parameters:
- frame: the frame to draw the closet panel on
- closet_items: a list of closet items to display
- selected_index: the index of the selected item in the closet

Returns:
- frame: the frame with the closet panel drawn on it
- thumbnail_positions: a list of tuples containing the positions of the thumbnails in the closet panel
"""
def draw_closet_panel(frame, closet_items, selected_index):
    panel_width = CLOSET_PANEL_WIDTH
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
    panel[:] = (50, 50, 50)  

    spacing = 10
    current_y = spacing
    thumbnail_positions = []  

    for i, item in enumerate(closet_items):
        thumb = item['thumbnail']
        x_offset = (panel_width - thumb.shape[1]) // 2
        y_offset = current_y
        panel[y_offset:y_offset+thumb.shape[0], x_offset:x_offset+thumb.shape[1]] = thumb[..., :3]
        if i == selected_index:
            cv2.rectangle(panel, (x_offset, y_offset), (x_offset+thumb.shape[1], y_offset+thumb.shape[0]), (0, 255, 0), 3)
        thumbnail_positions.append((x_offset, y_offset, x_offset+thumb.shape[1], y_offset+thumb.shape[0]))
        current_y += thumb.shape[0] + spacing

    frame[0:frame.shape[0], 0:panel_width] = panel

    return frame, thumbnail_positions
