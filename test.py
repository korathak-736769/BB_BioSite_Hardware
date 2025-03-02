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
import customtkinter as ctk
import tkinter as tk
import threading
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")
fontpath = os.path.join(base_path, "SukhumvitSet-Bold.ttf")
global_font = ("SukhumvitSet-Bold", 14)
font = ImageFont.truetype(fontpath, 24)

shoulder_width_threshold = 0.17
mouth_shoulder_ratio_threshold = 0.04
is_calibrated = False
Landmark = namedtuple('Landmark', ['x', 'y'])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

screen_width = pg.size().width

app = ctk.CTk()
app.title("Biosite Office Syndrome")
app.geometry("480x320")
app.resizable(False, False)
container = ctk.CTkFrame(app, width=480, height=320)
container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
menu_frame = ctk.CTkFrame(container)
calibrate_frame = ctk.CTkFrame(container)
monitoring_frame = ctk.CTkFrame(container)
for frame in (menu_frame, calibrate_frame, monitoring_frame):
    frame.grid(row=0, column=0, sticky="nsew")

def show_frame(frame):
    frame.tkraise()

status_label = ctk.CTkLabel(menu_frame, text="", font=global_font)

cap_monitor = None
video_running = False
skip_frames = 2
pose = None
face_mesh = None
frame_queue = []
processed_frame = None
queue_lock = threading.Lock()

def initialize_models():
    global pose, face_mesh
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def start_monitoring():
    global is_calibrated
    if not is_calibrated:
        status_label.configure(text="กรุณา Calibrate ก่อนเริ่มใช้งาน")
        return
    show_frame(monitoring_frame)
    start_video()

menu_title = ctk.CTkLabel(menu_frame, text="เลือกเมนู", font=("SukhumvitSet-Bold", 24))
menu_title.pack(pady=10)
calib_btn_menu = ctk.CTkButton(menu_frame, text="Calibrate", font=global_font, command=lambda: show_frame(calibrate_frame))
calib_btn_menu.pack(pady=5)
start_btn_menu = ctk.CTkButton(menu_frame, text="Start Monitoring", font=global_font, command=start_monitoring)
start_btn_menu.pack(pady=5)
status_label.pack(pady=5)

back_btn_calib = ctk.CTkButton(calibrate_frame, text="Back", font=global_font, command=lambda: show_frame(menu_frame))
back_btn_calib.pack(anchor="nw", padx=10, pady=5)
calib_title = ctk.CTkLabel(calibrate_frame, text="Calibrate Thresholds", font=("SukhumvitSet-Bold", 20))
calib_title.pack(pady=5)

manual_frame = ctk.CTkFrame(calibrate_frame)
manual_frame.pack(pady=5)
s_label = ctk.CTkLabel(manual_frame, text="Shoulder Threshold:", font=global_font)
s_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
shoulder_entry = ctk.CTkEntry(manual_frame, width=100, font=global_font)
shoulder_entry.grid(row=0, column=1, padx=5, pady=5)
m_label = ctk.CTkLabel(manual_frame, text="Mouth-Shoulder Threshold:", font=global_font)
m_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
mouth_entry = ctk.CTkEntry(manual_frame, width=100, font=global_font)
mouth_entry.grid(row=1, column=1, padx=5, pady=5)

countdown_label = ctk.CTkLabel(calibrate_frame, text="", font=global_font)
countdown_label.pack(pady=5)
loading_label = ctk.CTkLabel(calibrate_frame, text="", font=global_font)
calib_status_label = ctk.CTkLabel(calibrate_frame, text="", font=global_font)
calib_status_label.pack(pady=5)

loading_animation_running = False

def animate_loading():
    global loading_animation_running
    dots = 0
    def update():
        nonlocal dots
        if not loading_animation_running:
            return
        text = "Loading" + "." * (dots % 4)
        loading_label.configure(text=text)
        dots += 1
        calibrate_frame.after(500, update)
    update()

def wait_for_camera(cap, callback):
    if cap.isOpened():
        callback()
    else:
        calibrate_frame.after(100, lambda: wait_for_camera(cap, callback))

def start_countdown(t, cap):
    if t > 0:
        countdown_label.configure(text=str(t))
        calibrate_frame.after(1000, lambda: start_countdown(t - 1, cap))
    else:
        countdown_label.configure(text="")
        auto_calibrate(cap)

def load_calibration():
    global shoulder_width_threshold, mouth_shoulder_ratio_threshold, is_calibrated
    try:
        with open("calibration.json", "r") as f:
            data = json.load(f)
            shoulder_width_threshold = data.get("shoulder_width_threshold", 0.17)
            mouth_shoulder_ratio_threshold = data.get("mouth_shoulder_ratio_threshold", 0.04)
            is_calibrated = True
            calib_status_label.configure(text=f"Loaded calibration:\nShoulder {shoulder_width_threshold:.3f}, Mouth {mouth_shoulder_ratio_threshold:.3f}")
            shoulder_entry.delete(0, "end")
            shoulder_entry.insert(0, f"{shoulder_width_threshold:.3f}")
            mouth_entry.delete(0, "end")
            mouth_entry.insert(0, f"{mouth_shoulder_ratio_threshold:.3f}")
    except FileNotFoundError:
        print("Calibration file not found. Please calibrate.")

def save_calibration():
    data = {"shoulder_width_threshold": shoulder_width_threshold, "mouth_shoulder_ratio_threshold": mouth_shoulder_ratio_threshold}
    with open("calibration.json", "w") as f:
        json.dump(data, f)

def auto_calibrate(cap):
    calib_samples = 30
    shoulder_samples = []
    mouth_samples = []
    calib_pose = mp_pose.Pose(min_detection_confidence=0.5)
    calib_face_m = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    def capture_frame(count):
        ret, frame = cap.read()
        if not ret:
            calibrate_frame.after(10, lambda: capture_frame(count))
            return
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        pose_results = calib_pose.process(frame_rgb)
        face_mesh_results = calib_face_m.process(frame_rgb)
        if pose_results.pose_landmarks and face_mesh_results.multi_face_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            mouth_landmarks = [face_landmarks.landmark[i] for i, _ in mp_face_mesh.FACEMESH_LIPS]
            mouth_coords = np.mean([(lm.x, lm.y) for lm in mouth_landmarks], axis=0)
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * frame.shape[1]
            shoulder_ratio = shoulder_width / screen_width
            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y) / 2 - mouth_coords[1]) * frame.shape[0]
            mouth_ratio = mouth_shoulder_dist / screen_width
            shoulder_samples.append(shoulder_ratio)
            mouth_samples.append(mouth_ratio)
            count += 1
        if count < calib_samples:
            calibrate_frame.after(10, lambda: capture_frame(count))
        else:
            cap.release()
            calib_pose.close()
            calib_face_m.close()
            avg_shoulder = np.mean(shoulder_samples) if shoulder_samples else 0.17
            avg_mouth = np.mean(mouth_samples) if mouth_samples else 0.04
            margin = 0.1
            global shoulder_width_threshold, mouth_shoulder_ratio_threshold, is_calibrated
            shoulder_width_threshold = avg_shoulder * (1 + margin)
            mouth_shoulder_ratio_threshold = avg_mouth * (1 - margin)
            is_calibrated = True
            save_calibration()
            calib_status_label.configure(text=f"Calibrate Complete:\nShoulder {shoulder_width_threshold:.3f}, Mouth {mouth_shoulder_ratio_threshold:.3f}")
            shoulder_entry.delete(0, "end")
            shoulder_entry.insert(0, f"{shoulder_width_threshold:.3f}")
            mouth_entry.delete(0, "end")
            mouth_entry.insert(0, f"{mouth_shoulder_ratio_threshold:.3f}")
            app.after(2000, lambda: show_frame(menu_frame))
    capture_frame(0)

def start_calibration():
    global loading_animation_running
    loading_animation_running = True
    loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    animate_loading()
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    def after_camera_ready():
        global loading_animation_running
        loading_animation_running = False
        loading_label.place_forget()
        start_countdown(3, cap)
    wait_for_camera(cap, after_camera_ready)

calib_start_btn = ctk.CTkButton(calibrate_frame, text="Start Calibration", font=global_font, command=start_calibration)
calib_start_btn.pack(pady=10)

back_btn_monitor = ctk.CTkButton(monitoring_frame, text="Back", font=global_font, command=lambda: stop_video())
back_btn_monitor.pack(anchor="nw", padx=0, pady=0)
video_label = ctk.CTkLabel(monitoring_frame, text="", font=global_font)
video_label.pack()

def process_frame_thread():
    global cap_monitor, video_running, processed_frame, frame_queue
    last_text = "กำลังตรวจจับ..."
    last_color = (255, 255, 255)
    bad_start = None
    frame_count = 0
    while video_running:
        with queue_lock:
            if not frame_queue:
                time.sleep(0.001)
                continue
            frame = frame_queue.pop(0)
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        pose_results = pose.process(frame_rgb)
        face_mesh_results = face_mesh.process(frame_rgb)
        frame_disp = frame_rgb.copy()
        frame_disp.flags.writeable = True
        if pose_results.pose_landmarks and face_mesh_results.multi_face_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            mouth_landmarks_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
            mouth_coords = np.mean([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in mouth_landmarks_indices], axis=0)
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * frame_disp.shape[1]
            shoulder_ratio = shoulder_width / screen_width
            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y) / 2 - mouth_coords[1]) * frame_disp.shape[0]
            mouth_ratio = mouth_shoulder_dist / screen_width
            if shoulder_ratio > shoulder_width_threshold or mouth_ratio < mouth_shoulder_ratio_threshold:
                if bad_start is None:
                    bad_start = time.time()
                elapsed = time.time() - bad_start
                if elapsed >= 15:
                    last_text = "กรุณาปรับท่านั่งของคุณ"
                    last_color = (255, 0, 0)
                else:
                    last_text = f"ท่านั่งไม่เหมาะสม นับถอยหลัง: {15 - int(elapsed)}"
                    last_color = (255, 128, 0)
            else:
                last_text = "ท่านั่งของคุณเหมาะสมแล้ว"
                last_color = (0, 255, 0)
                bad_start = None
            cv.putText(frame_disp, f'Shoulder Ratio: {shoulder_ratio:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv.putText(frame_disp, f'Mouth-Shoulder Ratio: {mouth_ratio:.2f}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            connection_spec = mp_drawing.DrawingSpec(thickness=1)
            mp_drawing.draw_landmarks(frame_disp, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=connection_spec)
        pil_img = Image.new("RGBA", (500, 125), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        draw.text((100, 0), last_text, font=font, fill=last_color)
        text_img = np.array(pil_img)
        if text_img.shape[2] == 4:
            alpha_s = text_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            h, w = text_img.shape[:2]
            x = (frame_disp.shape[1] - w) // 2
            y = frame_disp.shape[0] - h - 25
            for c_ in range(3):
                frame_disp[y:y+h, x:x+w, c_] = alpha_s * text_img[:, :, c_] + alpha_l * frame_disp[y:y+h, x:x+w, c_]
        with queue_lock:
            processed_frame = frame_disp

def capture_frame_thread():
    global cap_monitor, video_running, frame_queue
    while video_running and cap_monitor is not None:
        ret, frame = cap_monitor.read()
        if ret:
            with queue_lock:
                frame_queue = [frame]
        else:
            time.sleep(0.001)

def update_ui():
    global processed_frame, video_running
    if not video_running:
        return
    with queue_lock:
        if processed_frame is not None:
            pil_frame = Image.fromarray(processed_frame)
            video_img = ctk.CTkImage(light_image=pil_frame, size=(pil_frame.width, pil_frame.height))
            video_label.configure(image=video_img)
            video_label.image = video_img
    if video_running:
        app.after(33, update_ui)

def start_video():
    global cap_monitor, video_running, frame_queue, processed_frame
    initialize_models()
    frame_queue = []
    processed_frame = None
    cap_monitor = cv.VideoCapture(1)
    cap_monitor.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap_monitor.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap_monitor.set(cv.CAP_PROP_FPS, 30)
    cap_monitor.set(cv.CAP_PROP_BUFFERSIZE, 1)
    video_running = True
    threading.Thread(target=capture_frame_thread, daemon=True).start()
    threading.Thread(target=process_frame_thread, daemon=True).start()
    update_ui()

def stop_video():
    global cap_monitor, video_running, pose, face_mesh
    video_running = False
    time.sleep(0.1)
    if cap_monitor is not None:
        cap_monitor.release()
        cap_monitor = None
    if pose is not None:
        pose.close()
    if face_mesh is not None:
        face_mesh.close()
    show_frame(menu_frame)

load_calibration()
show_frame(menu_frame)
app.mainloop()
