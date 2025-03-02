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

# ลดการแสดงข้อความ warning จาก TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

# ตั้งค่าพาธไฟล์
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")
fontpath = os.path.join(base_path, "SukhumvitSet-Bold.ttf")
global_font = ("SukhumvitSet-Bold", 16)
font = ImageFont.truetype(fontpath, 24)

# ตัวแปรสำหรับการ calibrate
shoulder_width_threshold = 0.17
mouth_shoulder_ratio_threshold = 0.04
is_calibrated = False
Landmark = namedtuple('Landmark', ['x', 'y'])

# ตั้งค่า MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# ขนาดหน้าจอ
screen_width = pg.size().width

# ตั้งค่า UI
app = ctk.CTk()
app.title("Biosite Office Syndrome")
app.geometry("800x600")
container = ctk.CTkFrame(app)
container.pack(fill="both", expand=True)
menu_frame = ctk.CTkFrame(container)
calibrate_frame = ctk.CTkFrame(container)
monitoring_frame = ctk.CTkFrame(container)
for frame in (menu_frame, calibrate_frame, monitoring_frame):
    frame.grid(row=0, column=0, sticky="nsew")

def show_frame(frame):
    frame.tkraise()

status_label = ctk.CTkLabel(menu_frame, text="", font=global_font)

# ตัวแปรสำหรับการทำงาน
cap_monitor = None
video_running = False
skip_frames = 2
pose = None
face_mesh = None
frame_queue = []
processed_frame = None
queue_lock = threading.Lock()

# สร้าง instance ของ MediaPipe แค่ครั้งเดียว
def initialize_models():
    global pose, face_mesh
    pose = mp_pose.Pose(
        static_image_mode=False,  # real-time performance
        model_complexity=1,       # เลือกระดับความซับซ้อนที่เหมาะสม (0, 1, 2)
        smooth_landmarks=True,    # ลดการสั่นของ landmark
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # real-time performance
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def start_monitoring():
    global is_calibrated
    if not is_calibrated:
        status_label.configure(text="กรุณา Calibrate ก่อนเริ่มใช้งาน")
        return
    show_frame(monitoring_frame)
    start_video()

menu_title = ctk.CTkLabel(menu_frame, text="เลือกเมนู", font=("SukhumvitSet-Bold", 24))
menu_title.pack(pady=20)
calib_btn_menu = ctk.CTkButton(menu_frame, text="Calibrate", font=global_font, command=lambda: show_frame(calibrate_frame))
calib_btn_menu.pack(pady=10)
start_btn_menu = ctk.CTkButton(menu_frame, text="Start Monitoring", font=global_font, command=start_monitoring)
start_btn_menu.pack(pady=10)
status_label.pack(pady=10)
back_btn_calib = ctk.CTkButton(calibrate_frame, text="Back", font=global_font, command=lambda: show_frame(menu_frame))
back_btn_calib.pack(anchor="nw", padx=10, pady=5)
calib_title = ctk.CTkLabel(calibrate_frame, text="Calibrate Thresholds", font=("SukhumvitSet-Bold", 20))
calib_title.pack(pady=10)
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
countdown_label.pack(pady=10)
calib_status_label = ctk.CTkLabel(calibrate_frame, text="", font=global_font)
calib_status_label.pack(pady=5)

def start_countdown(t):
    if t > 0:
        countdown_label.configure(text=str(t))
        calibrate_frame.after(1000, lambda: start_countdown(t - 1))
    else:
        auto_calibrate()

def auto_calibrate():
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    # ปรับการตั้งค่ากล้องเพื่อประสิทธิภาพ
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # ลดการหน่วงของภาพ
    
    calib_samples = 30
    shoulder_samples = []
    mouth_samples = []
    
    # สร้าง MediaPipe models สำหรับการ calibrate
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
            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y)/2 - mouth_coords[1]) * frame.shape[0]
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
            
            calib_status_label.configure(text=f"Calibrate Complete: Shoulder {shoulder_width_threshold:.3f}, Mouth {mouth_shoulder_ratio_threshold:.3f}")
            countdown_label.configure(text="")
            app.after(2000, lambda: show_frame(menu_frame))
    
    capture_frame(0)

def start_calibration():
    s_val = shoulder_entry.get().strip()
    m_val = mouth_entry.get().strip()
    
    if s_val and m_val:
        try:
            s_float = float(s_val)
            m_float = float(m_val)
            
            global shoulder_width_threshold, mouth_shoulder_ratio_threshold, is_calibrated
            shoulder_width_threshold = s_float
            mouth_shoulder_ratio_threshold = m_float
            is_calibrated = True
            
            calib_status_label.configure(text=f"Calibrate Complete: Shoulder {shoulder_width_threshold:.3f}, Mouth {mouth_shoulder_ratio_threshold:.3f}")
            app.after(2000, lambda: show_frame(menu_frame))
        except:
            calib_status_label.configure(text="กรุณาใส่ตัวเลขที่ถูกต้อง")
    else:
        countdown_label.configure(text="เตรียมนั่งในท่าที่ถูกต้อง")
        calibrate_frame.after(500, lambda: start_countdown(3))

calib_start_btn = ctk.CTkButton(calibrate_frame, text="Start Calibration", font=global_font, command=start_calibration)
calib_start_btn.pack(pady=10)
back_btn_monitor = ctk.CTkButton(monitoring_frame, text="Back", font=global_font, command=lambda: stop_video())
back_btn_monitor.pack(anchor="nw", padx=10, pady=5)
video_label = ctk.CTkLabel(monitoring_frame, text="", font=global_font)
video_label.pack()

# ฟังก์ชั่นสำหรับการประมวลผลเฟรมแบบแยกเธรด
def process_frame_thread():
    global cap_monitor, video_running, processed_frame, frame_queue
    
    last_text = "กำลังตรวจจับ..."
    last_color = (255, 255, 255)
    bad_start = None
    frame_count = 0
    
    while video_running:
        # ตรวจสอบว่ามีเฟรมใหม่ในคิวหรือไม่
        with queue_lock:
            if not frame_queue:
                time.sleep(0.001)  # เพื่อไม่ให้ CPU ทำงานหนักเกินไป
                continue
            frame = frame_queue.pop(0)
        
        # ข้ามเฟรมบางส่วนเพื่อเพิ่มประสิทธิภาพ
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        # เริ่มประมวลผลเฟรม
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # ประมวลผล MediaPipe
        pose_results = pose.process(frame_rgb)
        face_mesh_results = face_mesh.process(frame_rgb)
        
        frame_disp = frame_rgb.copy()
        frame_disp.flags.writeable = True
        
        if pose_results.pose_landmarks and face_mesh_results.multi_face_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            
            # ใช้เฉพาะ landmarks ที่จำเป็นสำหรับริมฝีปาก
            mouth_landmarks_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
            mouth_coords = np.mean([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in mouth_landmarks_indices], axis=0)
            
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * frame_disp.shape[1]
            shoulder_ratio = shoulder_width / screen_width
            mouth_shoulder_dist = abs((left_shoulder.y + right_shoulder.y) / 2 - mouth_coords[1]) * frame_disp.shape[0]
            mouth_ratio = mouth_shoulder_dist / screen_width
            
            # ตรวจสอบท่านั่ง
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
            
            # แสดงข้อมูลอัตราส่วน
            cv.putText(frame_disp, f'Shoulder Ratio: {shoulder_ratio:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv.putText(frame_disp, f'Mouth-Shoulder Ratio: {mouth_ratio:.2f}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # วาด landmarks ลงบนเฟรม
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            connection_spec = mp_drawing.DrawingSpec(thickness=1)
            mp_drawing.draw_landmarks(
                frame_disp, 
                pose_results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=connection_spec
            )
        
        # สร้างภาพข้อความ
        pil_img = Image.new("RGBA", (500, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), last_text, font=font, fill=last_color)
        text_img = np.array(pil_img)
        
        # ผสานภาพข้อความกับเฟรม
        if text_img.shape[2] == 4:
            alpha_s = text_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            h, w = text_img.shape[:2]
            x = (frame_disp.shape[1] - w) // 2
            y = frame_disp.shape[0] - h - 25
            for c_ in range(3):
                frame_disp[y:y+h, x:x+w, c_] = alpha_s * text_img[:, :, c_] + alpha_l * frame_disp[y:y+h, x:x+w, c_]
        
        # บันทึกเฟรมที่ประมวลผลแล้ว
        with queue_lock:
            processed_frame = frame_disp

# ฟังก์ชั่นสำหรับการดึงเฟรมจากกล้องแบบแยกเธรด
def capture_frame_thread():
    global cap_monitor, video_running, frame_queue
    
    while video_running and cap_monitor is not None:
        ret, frame = cap_monitor.read()
        if ret:
            with queue_lock:
                # เก็บเฉพาะเฟรมล่าสุด
                frame_queue = [frame]
        else:
            time.sleep(0.001)

# ฟังก์ชั่นสำหรับการอัพเดตหน้าจอ UI
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
        app.after(33, update_ui)  # ประมาณ 30 FPS

def start_video():
    global cap_monitor, video_running, frame_queue, processed_frame
    
    # เตรียม MediaPipe models
    initialize_models()
    
    # รีเซ็ตตัวแปรที่เกี่ยวข้อง
    frame_queue = []
    processed_frame = None
    
    # เปิดกล้อง
    cap_monitor = cv.VideoCapture(1)
    cap_monitor.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap_monitor.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap_monitor.set(cv.CAP_PROP_FPS, 30)
    cap_monitor.set(cv.CAP_PROP_BUFFERSIZE, 1)  # ลดการหน่วงของภาพ
    
    video_running = True
    
    # เริ่มเธรดสำหรับการจับภาพและประมวลผล
    threading.Thread(target=capture_frame_thread, daemon=True).start()
    threading.Thread(target=process_frame_thread, daemon=True).start()
    
    # เริ่มการอัพเดต UI
    update_ui()

def stop_video():
    global cap_monitor, video_running, pose, face_mesh
    
    video_running = False
    time.sleep(0.1)  # รอให้เธรดหยุด
    
    if cap_monitor is not None:
        cap_monitor.release()
        cap_monitor = None
    
    if pose is not None:
        pose.close()
    
    if face_mesh is not None:
        face_mesh.close()
    
    show_frame(menu_frame)

show_frame(menu_frame)
app.mainloop()