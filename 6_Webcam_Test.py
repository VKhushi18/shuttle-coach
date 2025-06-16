import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2

st.title("🎥 Webcam Test")

class BasicTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.putText(img, "✅ Webcam Running", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

webrtc_streamer(
    key="webcam-test",
    video_transformer_factory=BasicTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
) 