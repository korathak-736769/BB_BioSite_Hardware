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
import queue  # ใช้สำหรับ thread-safe queue

# เพิ่ม import สำหรับ Pi Camera
from picamera2 import Picamera2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# โหลดฟอนต์และโลโก้ (สมมุติว่ามีไฟล์ logo.png อยู่ในโฟลเดอร์เดียวกัน)
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
app.attributes("-fullscreen", True)
app.geometry("480x320")
app.resizable(False, False)

# สร้าง container หลักสำหรับวาง frame ต่างๆ
container = ctk.CTkFrame(app, width=480, height=320)
container.grid(row=0, column=0, sticky="nsew")

app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# สร้าง frame ทั้ง 4 หน้า: welcome, menu, calibrate และ monitoring
welcome_frame = ctk.CTkFrame(container, width=480, height=320)
menu_frame = ctk.CTkFrame(container, width=480, height=320)
calibrate_frame = ctk.CTkFrame(container, width=480, height=320)
monitoring_frame = ctk.CTkFrame(container, width=480, height=320)

welcome_frame.grid(row=0, column=0, sticky="nsew")
menu_frame.grid(row=0, column=0, sticky="nsew")
calibrate_frame.grid(row=0, column=0, sticky="nsew")
monitoring_frame.grid(row=0, column=0, sticky="nsew")

def show_frame(frame):
    frame.tkraise()

# ----------------- หน้า Welcome -----------------
welcome_inner = ctk.CTkFrame(welcome_frame)
welcome_inner.pack(expand=True)

if logo_image:
    logo_label = ctk.CTkLabel(welcome_inner, image=logo_image, text="")
    logo_label.pack(pady=10)
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

# ----------------- หน้า Menu -----------------
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

# ----------------- หน้า Calibrate Thresholds -----------------
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

        # ไม่ต้องแปลงสีอีก เพราะกล้องส่งออกเป็น RGB888 แล้ว
        frame_rgb = frame.copy()
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
            camera.stop()  # ปิดกล้องหลัง calibrate เสร็จ
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
            shoulder_entry.delete(0, "end")
            shoulder_entry.insert(0, f"{shoulder_width_threshold:.3f}")
            mouth_entry.delete(0, "end")
            mouth_entry.insert(0, f"{mouth_shoulder_ratio_threshold:.3f}")

            app.after(2000, lambda: show_frame(menu_frame))

    capture_frame(0)

def start_calibration():
    global loading_animation_running, picam2_calib
    loading_animation_running = True
    loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    animate_loading()

    # เปิดกล้อง Pi Camera สำหรับ calibration โดยใช้ RGB888
    picam2_calib = Picamera2()
    picam2_calib.configure(picam2_calib.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2_calib.start()

    # สมมุติว่ากล้องพร้อมใช้งานแล้ว
    loading_animation_running = False
    loading_label.place_forget()
    start_countdown(3, picam2_calib)

calib_start_btn = ctk.CTkButton(
    calibrate_frame,
    text="Start Calibration",
    font=global_font,
    command=start_calibration
)
calib_start_btn.pack(pady=10)

# ----------------- หน้า Monitoring -----------------
back_btn_monitor = ctk.CTkButton(
    monitoring_frame,
    text="Back",
    font=global_font,
    command=lambda: stop_video()
)
back_btn_monitor.pack(anchor="nw", padx=10, pady=5)

video_label = ctk.CTkLabel(monitoring_frame, text="", font=global_font)
video_label.pack(expand=True)

# ตัวแปรสำหรับการใช้ Pi Camera ในโหมด Monitoring
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
    last_color = (255, 255, 255)
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

        # ไม่ต้องแปลงสีอีก เพราะกล้องส่งออกเป็น RGB888 แล้ว
        frame_rgb = frame.copy()
        frame_rgb.flags.writeable = False

        pose_results = pose.process(frame_rgb)
        face_mesh_results = face_mesh.process(frame_rgb)

        frame_disp = frame_rgb.copy()
        frame_disp.flags.writeable = True

        if pose_results.pose_landmarks and face_mesh_results.multi_face_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            face_landmarks = face_mesh_results.multi_face_landmarks[0]

            mouth_landmarks_indices = [
                61, 185, 40, 39, 37, 0, 267, 269,
                270, 409, 291, 375, 321, 405, 314,
                17, 84, 181, 91, 146
            ]
            mouth_coords = np.mean(
                [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                 for i in mouth_landmarks_indices],
                axis=0
            )

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

            cv.putText(frame_disp, f'Shoulder Ratio: {shoulder_ratio:.2f}', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(frame_disp, f'Mouth-Shoulder Ratio: {mouth_ratio:.2f}', (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            connection_spec = mp_drawing.DrawingSpec(thickness=1)
            mp_drawing.draw_landmarks(
                frame_disp,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=connection_spec
            )

        # วาดข้อความลงในเฟรมโดยใช้ PIL
        pil_img = Image.new("RGBA", (500, 125), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        draw.text((100, 0), last_text, font=font, fill=last_color)
        text_img = np.array(pil_img)

        if text_img.shape[2] == 4:
            alpha_s = text_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            h_text, w_text = text_img.shape[:2]
            x = (frame_disp.shape[1] - w_text) // 2
            y = frame_disp.shape[0] - h_text - 25

            for c_ in range(3):
                frame_disp[y:y+h_text, x:x+w_text, c_] = (
                    alpha_s * text_img[:, :, c_] +
                    alpha_l * frame_disp[y:y+h_text, x:x+w_text, c_]
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
        
        # ใช้ขนาดคงที่ 480x320 (หรือใช้ video_label.winfo_width()/winfo_height())
        w = 480
        h = 320
        
        video_img = ctk.CTkImage(light_image=pil_frame, size=(w, h))
        video_label.configure(image=video_img)
        video_label.image = video_img
    if video_running:
        app.after(33, update_ui)

def start_video():
    global picam2_monitor, video_running, processed_frame
    initialize_models()
    processed_frame = None

    picam2_monitor = Picamera2()
    picam2_monitor.configure(picam2_monitor.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2_monitor.start()

    video_running = True
    threading.Thread(target=capture_frame_thread, daemon=True).start()
    threading.Thread(target=process_frame_thread, daemon=True).start()
    update_ui()

def stop_video():
    global picam2_monitor, video_running, pose, face_mesh
    video_running = False
    time.sleep(0.1)
    # ล้างคิวเพื่อให้ thread ไม่ถูก block ด้วยข้อมูลเก่า
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    if picam2_monitor is not None:
        picam2_monitor.stop()
        picam2_monitor = None
    if pose is not None:
        pose.close()
    if face_mesh is not None:
        face_mesh.close()
    show_frame(menu_frame)

load_calibration()
show_frame(welcome_frame)
app.mainloop()
