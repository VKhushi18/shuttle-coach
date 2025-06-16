# streamlit: hide
import streamlit as st
import cv2
import time
import numpy as np
import mediapipe as mp
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Hide the code by default
st.set_page_config(page_title="Badminton AI Game Mode", layout="wide")
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    .stApp header {visibility: hidden;}
    .stApp footer {visibility: hidden;}
    .stApp .viewerBadge_container__1QSob {display: none;}
    .stApp .stDeployButton {display: none;}
    .stApp .stApp header {visibility: hidden;}
    .stApp .stApp footer {visibility: hidden;}
    .stApp .stApp .viewerBadge_container__1QSob {display: none;}
    .stApp .stApp .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-top: 1rem;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .chart-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .code-container {
        display: none;
    }
    .feedback-container {
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 15px;
        border-radius: 10px;
        z-index: 1000;
        max-width: 300px;
    }
</style>
""", unsafe_allow_html=True)

# Main title with emoji
st.markdown('<h1 class="main-header">🏸 AI-Powered Badminton Game Mode</h1>', unsafe_allow_html=True)

# Add Performance Analysis Button
col1, col2, col3 = st.columns([1, 2, 1])
with col3:
    if st.button("📊 Performance Timeline", use_container_width=True):
        st.switch_page("pages/6_Performance_Timeline.py")

# Sidebar settings with improved styling
st.sidebar.markdown("""
<div style='background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;'>
    <h2 style='color: #4CAF50;'>⚙️ Training Settings</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar settings with descriptions
st.sidebar.markdown("### 🎯 Session Configuration")
max_events = st.sidebar.slider(
    "Number of Shuttles", 
    min_value=5, 
    max_value=30, 
    value=10,
    help="Set how many shuttles you want to practice with in this session"
)

session_duration = st.sidebar.slider(
    "Session Duration (seconds)", 
    min_value=10, 
    max_value=300, 
    value=60,
    help="Set the maximum duration for your training session"
)

# Feedback settings
st.sidebar.markdown("### 🎯 Feedback Settings")
show_pose_overlay = st.sidebar.checkbox("Show Pose Overlay", value=True, help="Display skeleton overlay on your body")
show_reaction_time = st.sidebar.checkbox("Show Reaction Time", value=True, help="Display your reaction time in real-time")
show_shuttle_tracking = st.sidebar.checkbox("Show Shuttle Tracking", value=True, help="Highlight detected shuttles")

# Mode selection with improved styling
st.markdown('<h2 class="sub-header">🎥 Choose Your Training Mode</h2>', unsafe_allow_html=True)
mode = st.radio(
    "Select how you want to practice:",
    ["📹 Upload a rally video", "📵 Use webcam (real-time)"],
    horizontal=True,
    help="Choose between uploading a video or using your webcam for real-time training"
)

# Instructions based on selected mode
if mode == "📹 Upload a rally video":
    st.markdown("""
    <div class="info-box">
        <h3>📝 Video Upload Instructions:</h3>
        <ol>
            <li>Upload a video of your badminton rally</li>
            <li>The AI will analyze your technique and posture</li>
            <li>You'll receive feedback on your performance</li>
            <li>A comprehensive report will be generated at the end</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
        <h3>📝 Webcam Instructions:</h3>
        <ol>
            <li>Position yourself in front of the camera</li>
            <li>Make sure your full body is visible</li>
            <li>Practice your badminton movements</li>
            <li>The AI will provide real-time feedback</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

class ShuttleTrainer:
    def __init__(self, max_events):
        self.reactions = []
        self.correct = 0
        self.mistake = 0
        self.event_index = 0
        self.max_events = max_events
        self.wait_time = 3
        self.last_event_time = 0
        self.start_timer = False
        self.visual_log = []
        self.last_shuttle_time = 0
        self.shuttle_missed = False
        self.missed_count = 0
        self.feedback_messages = []
        self.current_feedback = ""
        self.feedback_duration = 0
        self.feedback_start_time = 0

    def add_feedback(self, message, duration=3):
        self.current_feedback = message
        self.feedback_start_time = time.time()
        self.feedback_duration = duration
        self.feedback_messages.append((message, time.time()))

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24") if isinstance(frame, av.VideoFrame) else frame
        h, w, _ = image.shape

        # Add session info at the top
        cv2.putText(image, f"Session: {self.event_index}/{self.max_events} shuttles", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timer if session is active
        if self.start_timer:
            elapsed = time.time() - self.last_event_time
            cv2.putText(image, f"Time: {elapsed:.1f}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.event_index >= self.max_events:
            cv2.putText(image, f"🏁 Session Finished", (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            return image

        # Detect shuttles
        results = model.predict(source=image, conf=0.4, classes=[0], verbose=False)
        shuttle_detected = any(len(r.boxes) > 0 for r in results)
        
        # Draw shuttle bounding boxes if enabled
        if show_shuttle_tracking:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"Shuttle: {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check for missed shuttles
        current_time = time.time()
        if shuttle_detected:
            self.last_shuttle_time = current_time
            self.shuttle_missed = False
        elif current_time - self.last_shuttle_time > 2 and not self.shuttle_missed and self.last_shuttle_time > 0:
            self.shuttle_missed = True
            self.missed_count += 1
            self.add_feedback("❌ Shuttle missed! Try to keep your eye on it", 2)

        if shuttle_detected:
            if not self.start_timer:
                self.last_event_time = time.time()
                self.start_timer = True
                self.add_feedback("⏒ React!", 1)

            elapsed = time.time() - self.last_event_time
            cv2.putText(image, f"⏒ React!", (w//2 - 100, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            # Process pose
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            correct_posture = False

            if results.pose_landmarks:
                if show_pose_overlay:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                landmarks = results.pose_landmarks.landmark
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                # Check if hands are above hips (ready position)
                if left_hand.y < left_hip.y and right_hand.y < right_hip.y:
                    correct_posture = True
                else:
                    self.add_feedback("⚠️ Raise your hands!", 1)

            if elapsed > np.random.uniform(1.5, 3.0):
                reaction_time = elapsed
                self.reactions.append(reaction_time)
                feedback_text = "✅ Good form!" if correct_posture else "❌ Fix posture!"
                color = (0, 255, 0) if correct_posture else (0, 0, 255)
                
                # Add feedback message
                self.add_feedback(feedback_text, 2)
                
                # Display reaction time if enabled
                if show_reaction_time:
                    cv2.putText(image, f"Reaction: {reaction_time:.2f}s", (w//2 - 100, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if correct_posture:
                    self.correct += 1
                else:
                    self.mistake += 1
                self.visual_log.append((self.event_index + 1, reaction_time, feedback_text))

                self.start_timer = False
                self.event_index += 1
        else:
            cv2.putText(image, "🔍 Waiting for shuttle...", (w//2 - 150, 80), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Display current feedback message
        if self.current_feedback and time.time() - self.feedback_start_time < self.feedback_duration:
            # Create a semi-transparent overlay for feedback
            overlay = image.copy()
            cv2.rectangle(overlay, (w//2 - 200, h - 100), (w//2 + 200, h - 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Add feedback text
            cv2.putText(image, self.current_feedback, (w//2 - 180, h - 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display stats at the bottom
        cv2.putText(image, f"Correct: {self.correct} | Mistakes: {self.mistake} | Missed: {self.missed_count}", 
                  (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image

def show_report(trainer):
    st.markdown('<h2 class="sub-header">📊 Training Session Report</h2>', unsafe_allow_html=True)
    
    # Success message with animation
    st.success("✅ Session Complete! Great job!")
    
    # Metrics in a grid layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("✅ Correct Postures", trainer.correct)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("❌ Mistakes", trainer.mistake)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Missed Shuttles", trainer.missed_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if len(trainer.reactions) > 0:
            st.metric("⏱️ Avg Reaction Time", f"{np.mean(trainer.reactions):.2f} sec")
        else:
            st.metric("⏱️ Avg Reaction Time", "No data")
        st.markdown('</div>', unsafe_allow_html=True)

    df = pd.DataFrame(trainer.visual_log, columns=["Event", "ReactionTime", "Feedback"])

    # Pie chart with improved styling
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📊 Posture Performance")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    if trainer.correct == 0 and trainer.mistake == 0:
        ax3.pie([1], labels=["No Data"], colors=["gray"], autopct="%1.1f%%", startangle=90)
        ax3.set_title("Posture Accuracy (No Data Collected)")
    else:
        ax3.pie([trainer.correct, trainer.mistake], labels=["Correct", "Incorrect"], 
                colors=["green", "red"], autopct="%1.1f%%", startangle=90)
        ax3.set_title("Posture Accuracy")
    
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

    # Only show these charts if there's data
    if not df.empty:
        # Reaction timeline with improved styling
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Reaction Timeline")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df, x="Event", y="ReactionTime", hue="Feedback", palette="Set2", ax=ax)
        ax.set_ylabel("Reaction Time (s)")
        ax.set_title("Reaction Time by Event")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Distribution of reaction times with improved styling
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 Distribution of Reaction Times")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(df["ReactionTime"], kde=True, ax=ax2, color="skyblue")
        ax2.set_xlabel("Reaction Time (s)")
        ax2.set_title("Distribution of Reaction Times")
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feedback summary
    if trainer.feedback_messages:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("💬 Feedback Summary")
        
        # Count feedback types
        feedback_counts = {}
        for msg, _ in trainer.feedback_messages:
            if msg in feedback_counts:
                feedback_counts[msg] += 1
            else:
                feedback_counts[msg] = 1
        
        # Display feedback counts
        for msg, count in feedback_counts.items():
            st.markdown(f"- **{msg}**: {count} times")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Download button with improved styling
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("📥 Download Your Report")
    st.markdown("Save your training data for future reference and track your progress over time.")
    st.download_button("📊 Download Feedback CSV", data=df.to_csv(index=False), file_name="badminton_report.csv")
    st.markdown('</div>', unsafe_allow_html=True)

    # Add button to view Performance Timeline
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.subheader("📈 View Detailed Performance Timeline")
    st.markdown("Click below to see a detailed timeline of your performance metrics and progress.")
    if st.button("🕒 View Performance Timeline", key="view_timeline"):
        st.switch_page("pages/6_Performance_Timeline.py")
    st.markdown('</div>', unsafe_allow_html=True)

if mode == "📹 Upload a rally video":
    video_file = st.file_uploader("Upload a rally video", type=["mp4", "mov"])
    if video_file:
        st.video(video_file)
        trainer = ShuttleTrainer(max_events)
        with open("temp.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")
        stframe = st.empty()
        start = time.time()
        while cap.isOpened() and trainer.event_index < trainer.max_events and (time.time() - start) < session_duration:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame = trainer.transform(frame)
            stframe.image(out_frame, channels="BGR")
                
        cap.release()
        show_report(trainer)

elif mode == "📵 Use webcam (real-time)":
    trainer = ShuttleTrainer(max_events)

    if "show_report" not in st.session_state:
        st.session_state.show_report = False

    class LiveTrainer(VideoTransformerBase):
        def __init__(self):
            self.trainer = trainer
            self.start = time.time()

        def transform(self, frame):
            elapsed = time.time() - self.start
            if elapsed > session_duration or self.trainer.event_index >= self.trainer.max_events:
                st.session_state.show_report = True
                return frame.to_ndarray(format="bgr24")
            return self.trainer.transform(frame)

    # Webcam container with improved styling
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🎥 Live Training Session")
    st.markdown("Position yourself in front of the camera and start practicing. The AI will analyze your technique in real-time.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time feedback container
    feedback_container = st.empty()
    
    ctx = webrtc_streamer(
        key="game-mode",
        video_transformer_factory=LiveTrainer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True
    )

    if st.session_state.show_report:
        st.success("🎯 Webcam session finished. Generating report...")
        show_report(trainer) 