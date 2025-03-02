import cv2 as cv
import numpy as np
import os
import sys
import mediapipe as mp
import pyautogui as pg
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from collections import namedtuple
import logging
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

# if getattr(sys, 'frozen', False):
#     mp._framework_bindings.resource_utils.set_resource_dir(
#         os.path.join(sys._MEIPASS, 'mediapipe')
#     )

# command build : pyinstaller --noconfirm --onefile --windowed ^
# --add-data "SukhumvitSet-Bold.ttf;." ^
# --collect-data mediapipe ^
# --hidden-import "tensorflow.keras.engine" ^
# --hidden-import "mediapipe.resource_utils" ^
# your_script.py 

    
if getattr(sys, 'frozen', False):  
    base_path = sys._MEIPASS  
else:
    base_path = os.path.abspath(".") 

Landmark = namedtuple('Landmark', ['x', 'y'])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Constants
screen_width = pg.size().width
shoulder_width_threshold = 0.17  
mouth_shoulder_ratio_threshold = 0.04  
fontpath = os.path.join(base_path, "SukhumvitSet-Bold.ttf")
font = ImageFont.truetype(fontpath, 24)  

cap = cv.VideoCapture(1) 

shoulder_width_ratio = 0.0
mouth_shoulder_ratio = 0.0

bad_posture_start_time = None  

with mp_pose.Pose(min_detection_confidence=0.5) as pose, \
        mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Image preprocessing
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        pose_results = pose.process(image)
        face_mesh_results = face_mesh.process(image)

        image_copy = image.copy()  # Create a copy for drawing results

        if pose_results.pose_landmarks and face_mesh_results.multi_face_landmarks:
            # Landmark calculation
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            mouth_landmarks = [face_landmarks.landmark[i] for i, _ in mp_face_mesh.FACEMESH_LIPS] 
            mouth_x, mouth_y = np.mean([(lm.x, lm.y) for lm in mouth_landmarks], axis=0)
            mouth = Landmark(x=mouth_x, y=mouth_y)

            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image.shape[1]
            shoulder_width_ratio = shoulder_width / screen_width
            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y) / 2 - mouth.y) * image.shape[0]
            mouth_shoulder_ratio = mouth_shoulder_dist / screen_width

            # Check posture and set text
            if shoulder_width_ratio > shoulder_width_threshold or mouth_shoulder_ratio < mouth_shoulder_ratio_threshold:
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()

                elapsed_time = time.time() - bad_posture_start_time
                if elapsed_time >= 15:
                    text = "กรุณาปรับท่านั่งของคุณ"
                    color = (0, 0, 255)  # Red
                else:
                    text = f"ท่านั่งไม่เหมาะสม นับถอยหลัง: {15 - int(elapsed_time)}"
                    color = (0, 128, 255)  # Orange
            else:
                text = "ท่านั่งของคุณเหมาะสมแล้ว"
                color = (0, 255, 0)  # Green
                bad_posture_start_time = None

            # Render text
            text_img = Image.new('RGBA', (500, 100), (0, 0, 0, 0))  # Create blank image
            draw = ImageDraw.Draw(text_img)
            text_bbox = draw.textbbox((0, 0), text, font=font)  # Get text bounding box
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_img = Image.new('RGBA', (text_width + 20, text_height + 20), (0, 0, 0, 0))  # Resize image to fit text
            draw = ImageDraw.Draw(text_img)
            draw.text((10, 10), text, font=font, fill=color)  # Draw text on image

            # Convert text image to OpenCV format
            text_img = np.array(text_img)
            text_img = cv.cvtColor(text_img, cv.COLOR_RGBA2BGRA)

            # Overlay text on image
            h, w, _ = image_copy.shape
            text_img_h, text_img_w, _ = text_img.shape
            x = (w - text_img_w) // 2  # Center horizontally
            y = h - text_img_h - 25   # Place at the bottom

            alpha_s = text_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                image_copy[y:y + text_img_h, x:x + text_img_w, c] = (alpha_s * text_img[:, :, c] +
                                                                      alpha_l * image_copy[y:y + text_img_h, x:x + text_img_w, c])

        # Display information on the image
        cv.putText(image_copy, f'Shoulder Ratio: {shoulder_width_ratio:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(image_copy, f'Mouth-Shoulder Ratio: {mouth_shoulder_ratio:.2f}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show image
        image_copy.flags.writeable = True
        image_copy = cv.cvtColor(image_copy, cv.COLOR_RGB2BGR)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image_copy, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv.imshow('Biosite Office Syndrome', image_copy)
        if cv.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv.destroyAllWindows()