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
import queue

from picamera2 import Picamera2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

# กำหนดพาธไฟล์
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# ระบบฟอนต์และภาพ
fontpath = os.path.join(base_path, "SukhumvitSet-Bold.ttf")
global_font = ("SukhumvitSet-Bold", 14)
font = ImageFont.truetype(fontpath, 24)
logo_path = os.path.join(base_path, "./icon_title.png")

if os.path.exists(logo_path):
    from PIL import Image as PILImage
    pil_logo = PILImage.open(logo_path)
    pil_logo = pil_logo.resize((200, 200))
    logo_image = ctk.CTkImage(light_image=pil_logo, size=(200, 200))
else:
    logo_image = None

# ค่าคอนฟิกเริ่มต้น
shoulder_width_threshold = 0.17
mouth_shoulder_ratio_threshold = 0.04
is_calibrated = False
Landmark = namedtuple('Landmark', ['x', 'y'])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

screen_width = pg.size().width

# ระบบ GUI
app = ctk.CTk()
app.title("Biosite Office Syndrome")
app.attributes("-fullscreen", True)
app.geometry("480x320")
app.resizable(False, False)

container = ctk.CTkFrame(app, width=480, height=320)
container.grid(row=0, column=0, sticky="nsew")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# เฟรมต่างๆ
welcome_frame = ctk.CTkFrame(container, width=480, height=320)
menu_frame = ctk.CTkFrame(container, width=480, height=320)
calibrate_frame = ctk.CTkFrame(container, width=480, height=320)
monitoring_frame = ctk.CTkFrame(container, width=480, height=320)

for frame in [welcome_frame, menu_frame, calibrate_frame, monitoring_frame]:
    frame.grid(row=0, column=0, sticky="nsew")

def show_frame(frame):
    frame.tkraise()

# ----------------- Welcome Frame -----------------
welcome_inner = ctk.CTkFrame(welcome_frame)
welcome_inner.pack(expand=True)

if logo_image:
    logo_label = ctk.CTkLabel(welcome_inner, image=logo_image, text="")
else:
    logo_label = ctk.CTkLabel(welcome_inner, text="LOGO", font=("SukhumvitSet-Bold", 24))
logo_label.pack(pady=10)

start_usage_btn = ctk.CTkButton(
    welcome_inner,
    text="เริ่มต้นการใช้งาน",
    font=global_font,
    command=lambda: show_frame(menu_frame)
)
start_usage_btn.pack(pady=10)

# ----------------- Menu Frame -----------------
menu_inner = ctk.CTkFrame(menu_frame)
menu_inner.pack(expand=True)

menu_title = ctk.CTkLabel(menu_inner, text="เลือกเมนู", font=("SukhumvitSet-Bold", 24))
menu_title.pack(pady=10)

status_label = ctk.CTkLabel(menu_inner, text="", font=global_font)
status_label.pack(pady=5)

def start_monitoring():
    global is_calibrated
    if not is_calibrated:
        status_label.configure(text="กรุณา Calibrate ก่อนเริ่มใช้งาน")
        return
    show_frame(monitoring_frame)
    start_video()

calib_btn_menu = ctk.CTkButton(
    menu_inner, text="Calibrate",
    font=global_font,
    command=lambda: show_frame(calibrate_frame)
)
calib_btn_menu.pack(pady=5)

start_btn_menu = ctk.CTkButton(
    menu_inner,
    text="Start Monitoring",
    font=global_font,
    command=start_monitoring
)
start_btn_menu.pack(pady=5)

exit_btn_menu = ctk.CTkButton(
    menu_inner,
    text="ออกจากโปรแกรม",
    font=global_font,
    command=app.destroy
)
exit_btn_menu.pack(pady=5)

# ----------------- Calibrate Frame -----------------
back_btn_calib = ctk.CTkButton(
    calibrate_frame,
    text="Back",
    font=global_font,
    command=lambda: show_frame(menu_frame)
)
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

def animate_loading():
    dots = 0
    def update():
        nonlocal dots
        text = "Loading" + "." * (dots % 4)
        loading_label.configure(text=text)
        dots += 1
        calibrate_frame.after(500, update)
    update()

def start_countdown(t, camera):
    if t > 0:
        countdown_label.configure(text=str(t))
        calibrate_frame.after(1000, lambda: start_countdown(t - 1, camera))
    else:
        countdown_label.configure(text="")
        auto_calibrate(camera)

def load_calibration():
    global shoulder_width_threshold, mouth_shoulder_ratio_threshold, is_calibrated
    try:
        with open("calibration.json", "r") as f:
            data = json.load(f)
            shoulder_width_threshold = data.get("shoulder_width_threshold", 0.17)
            mouth_shoulder_ratio_threshold = data.get("mouth_shoulder_ratio_threshold", 0.04)
            is_calibrated = True
            calib_status_label.configure(
                text=f"Loaded calibration:\nShoulder {shoulder_width_threshold:.3f}, Mouth {mouth_shoulder_ratio_threshold:.3f}"
            )
            shoulder_entry.delete(0, "end")
            shoulder_entry.insert(0, f"{shoulder_width_threshold:.3f}")
            mouth_entry.delete(0, "end")
            mouth_entry.insert(0, f"{mouth_shoulder_ratio_threshold:.3f}")
    except FileNotFoundError:
        print("Calibration file not found. Please calibrate.")

def save_calibration():
    data = {
        "shoulder_width_threshold": shoulder_width_threshold,
        "mouth_shoulder_ratio_threshold": mouth_shoulder_ratio_threshold
    }
    with open("calibration.json", "w") as f:
        json.dump(data, f)

def auto_calibrate(camera):
    calib_samples = 30
    shoulder_samples = []
    mouth_samples = []
    calib_pose = mp_pose.Pose(min_detection_confidence=0.5)
    calib_face_m = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    def capture_frame(count):
        frame = camera.capture_array()
        if frame is None:
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

            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y)/2 - mouth_coords[1]) * frame.shape[0]
            mouth_ratio = mouth_shoulder_dist / screen_width

            shoulder_samples.append(shoulder_ratio)
            mouth_samples.append(mouth_ratio)
            count += 1

        if count < calib_samples:
            calibrate_frame.after(10, lambda: capture_frame(count))
        else:
            camera.stop()
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

            calib_status_label.configure(
                text=f"Calibrate Complete:\nShoulder {shoulder_width_threshold:.3f}, Mouth {mouth_shoulder_ratio_threshold:.3f}"
            )
            app.after(2000, lambda: show_frame(menu_frame))

    capture_frame(0)

def start_calibration():
    global picam2_calib
    if picam2_calib is not None:
        picam2_calib.stop()
        picam2_calib = None

    loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    animate_loading()

    picam2_calib = Picamera2()
    picam2_calib.configure(picam2_calib.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
    picam2_calib.start()

    loading_label.place_forget()
    start_countdown(3, picam2_calib)

calib_start_btn = ctk.CTkButton(
    calibrate_frame,
    text="Start Calibration",
    font=global_font,
    command=start_calibration
)
calib_start_btn.pack(pady=10)

# ----------------- Monitoring Frame -----------------
back_btn_monitor = ctk.CTkButton(
    monitoring_frame,
    text="Back",
    font=global_font,
    command=lambda: stop_video()
)
back_btn_monitor.pack(anchor="nw", padx=10, pady=5)

video_label = ctk.CTkLabel(monitoring_frame, text="", font=global_font)
video_label.pack(expand=True)

picam2_monitor = None
video_running = False
skip_frames = 2
pose = None
face_mesh = None
frame_queue = queue.Queue(maxsize=1)
processed_frame = None

def initialize_models():
    global pose, face_mesh
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_frame_thread():
    global video_running, processed_frame
    last_text = "กำลังตรวจจับ..."
    last_color = (0, 255, 0)  # BGR: เขียว
    bad_start = None
    frame_count = 0

    while video_running:
        try:
            frame = frame_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # แปลงสีสำหรับ MediaPipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        pose_results = pose.process(frame_rgb)
        face_mesh_results = face_mesh.process(frame_rgb)

        # แปลงกลับเป็น BGR สำหรับวาดภาพ
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        frame_bgr.flags.writeable = True

        if pose_results.pose_landmarks and face_mesh_results.multi_face_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            face_landmarks = face_mesh_results.multi_face_landmarks[0]

            mouth_landmarks_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
            mouth_coords = np.mean(
                [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) 
                 for i in mouth_landmarks_indices],
                axis=0
            )

            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * frame_bgr.shape[1]
            shoulder_ratio = shoulder_width / screen_width

            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y)/2 - mouth_coords[1]) * frame_bgr.shape[0]
            mouth_ratio = mouth_shoulder_dist / screen_width

            if shoulder_ratio > shoulder_width_threshold or mouth_ratio < mouth_shoulder_ratio_threshold:
                if bad_start is None:
                    bad_start = time.time()
                elapsed = time.time() - bad_start
                if elapsed >= 15:
                    last_text = "กรุณาปรับท่านั่งของคุณ"  # แก้ไขคำผิด
                    last_color = (0, 0, 255)  # BGR: แดง
                else:
                    last_text = f"ท่านั่งไม่เหมาะสม นับถอยหลัง: {15 - int(elapsed)}"
                    last_color = (0, 128, 255)  # BGR: ส้ม
            else:
                last_text = "ท่านั่งของคุณเหมาะสมแล้ว"
                last_color = (0, 255, 0)  # BGR: เขียว
                bad_start = None

            # วาดข้อมูลบนเฟรม
            cv.putText(frame_bgr, f'Shoulder Ratio: {shoulder_ratio:.2f}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(frame_bgr, f'Mouth-Shoulder Ratio: {mouth_ratio:.2f}', (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # วาด landmarks
            mp_drawing.draw_landmarks(
                frame_bgr,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(thickness=1)
            )

        # แปลงเป็น RGB สำหรับแสดงผล
        frame_disp = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

        # เพิ่มข้อความด้วย PIL
        pil_img = Image.new("RGBA", (500, 125), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        draw.text((100, 0), last_text, font=font, fill=last_color)
        text_img = np.array(pil_img)

        # ผสานภาพ
        if text_img.shape[2] == 4:
            alpha_s = text_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            h, w = text_img.shape[:2]
            x = (frame_disp.shape[1] - w) // 2
            y = frame_disp.shape[0] - h - 25

            for c in range(3):
                frame_disp[y:y+h, x:x+w, c] = (
                    alpha_s * text_img[:, :, c] + 
                    alpha_l * frame_disp[y:y+h, x:x+w, c]
                )

        processed_frame = frame_disp

def capture_frame_thread():
    global picam2_monitor, video_running
    while video_running and picam2_monitor is not None:
        frame = picam2_monitor.capture_array()
        if frame is not None:
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)
        else:
            time.sleep(0.001)

def update_ui():
    global processed_frame, video_running
    if not video_running:
        return
    if processed_frame is not None:
        pil_frame = Image.fromarray(processed_frame, 'RGB')
        video_img = ctk.CTkImage(light_image=pil_frame, size=(480, 320))
        video_label.configure(image=video_img)
        video_label.image = video_img
    app.after(33, update_ui)

def start_video():
    global picam2_monitor, video_running
    if picam2_monitor is not None:
        picam2_monitor.stop()
        picam2_monitor = None

    initialize_models()
    picam2_monitor = Picamera2()
    picam2_monitor.configure(picam2_monitor.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
    picam2_monitor.start()

    video_running = True
    threading.Thread(target=capture_frame_thread, daemon=True).start()
    threading.Thread(target=process_frame_thread, daemon=True).start()
    update_ui()

def stop_video():
    global picam2_monitor, video_running, pose, face_mesh
    video_running = False
    time.sleep(0.5)  # รอให้เธรดทำงานเสร็จ
    
    # ล้างคิว
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    
    # ปิดทรัพยากร
    if picam2_monitor is not None:
        picam2_monitor.stop()
        picam2_monitor.close()
        picam2_monitor = None
    
    if pose is not None:
        pose.close()
        pose = None
    
    if face_mesh is not None:
        face_mesh.close()
        face_mesh = None
    
    # รีเซ็ต UI
    video_label.configure(image=None)
    video_label.image = None
    show_frame(menu_frame)

load_calibration()
show_frame(welcome_frame)
app.mainloop()