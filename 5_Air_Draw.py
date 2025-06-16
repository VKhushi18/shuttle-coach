import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
from PIL import Image
import tempfile
import os

# Page setup
st.set_page_config(page_title="Badminton Air Draw", layout="wide")
st.title("🏸 Air Draw on Badminton Court")

# Upload court image
court_img_file = st.sidebar.file_uploader("Upload Badminton Court Image", type=["png", "jpg", "jpeg"])

# Load court image
if court_img_file is not None:
    img = Image.open(court_img_file)
    img = img.convert("RGB")
    court_image_np = np.array(img)
    court_image = cv2.resize(court_image_np, (640, 480))
else:
    st.warning("Please upload a badminton court image to start drawing.")
    court_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Drawing settings
st.sidebar.header("🖌️ Drawing Settings")
color_hex = st.sidebar.color_picker("Choose Drawing Color", "#FF0000")
thickness = st.sidebar.slider("Line Thickness", 1, 10, 4)
drawing_color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

# MediaPipe
mp_hands = mp.solutions.hands

class AirDraw(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing = False
        self.latest_output = None
        self.current_color = drawing_color
        self.current_thickness = thickness

    def update_settings(self, color, thickness):
        self.current_color = color
        self.current_thickness = thickness

    def is_pinch(self, landmarks):
        index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        return np.hypot(index.x - thumb.x, index.y - thumb.y) < 0.04

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        if court_img_file is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Update color and thickness settings
        self.update_settings(drawing_color, thickness)

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if self.is_pinch(hand_landmarks):
                    x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 640)
                    y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 480)

                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (x, y), self.current_color, self.current_thickness)
                    self.prev_point = (x, y)
                    self.drawing = True
                else:
                    self.prev_point = None
                    self.drawing = False

        # Combine court image and canvas
        display = court_image.copy()
        display = cv2.addWeighted(display, 1.0, self.canvas, 1.0, 0)

        self.latest_output = display.copy()
        return display

drawer_instance = AirDraw()

if court_img_file:
    webrtc_ctx = webrtc_streamer(
        key="airdraw",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: drawer_instance,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Save/export
    if st.button("💾 Export Drawing"):
        if drawer_instance.latest_output is not None:
            filename = "badminton_drawing.png"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            cv2.imwrite(filepath, drawer_instance.latest_output)
            with open(filepath, "rb") as file:
                st.download_button("📥 Download Drawing", file, file_name=filename)
        else:
            st.warning("Nothing to export yet.")

    # Clear canvas with confirmation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗑️ Clear Canvas")
    st.sidebar.markdown("Click the button below to clear your drawing.")
    
    if st.sidebar.button("Clear All Drawings", type="primary"):
        if st.sidebar.button("Confirm Clear", type="secondary"):
            drawer_instance.canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            drawer_instance.prev_point = None
            st.sidebar.success("Canvas cleared!")
            st.experimental_rerun()
        else:
            st.sidebar.info("Canvas not cleared. Click 'Confirm Clear' to proceed.") 