# Virtual Dressing Room

This project is a **virtual dressing room** application that allows users to try on **t-shirts in real-time** using **computer vision** and **pose detection**. The application is based on **MediaPipe** and **OpenCV** to accurately overlay garments on the user's torso, providing a smooth and interactive experience.

## Requirements

- **Python 3.11.x or lower** (due to compatibility issues with **MediaPipe** in newer versions).
- **Webcam** (required for real-time detection).
- **PIP** installed.

## Installation

### 1. Create a Virtual Environment (Optional)
It is recommended to create a virtual environment to avoid conflicts with other dependencies:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies
Install all required dependencies by running:

``` bash
pip install -r doc/requirements.txt
```
This will install **MediaPipe**, **OpenCV**, and other required libraries.

### 3. Run the Application
To start the virtual dressing room, execute:
```bash
python main.py
```

## Notes
* **Python 3.11.x or lower** is recommended to avoid conflicts with MediaPipe.
* Ensure that your **webcam** is enabled and functioning correctly.
* At this point, the app is only designed for working with `.png` files (i.e. no background) and fixed size, if you want to add a new shirt, it should **maintain the same dimensions** as the `Shirts/c_*.png` images.