import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, AudioFileClip, concatenate_audioclips
import random
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Highlight Reel Generator - AI ShuttleCoach",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1e1e1e;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1a1a1a;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #4fc3f7;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #b0bec5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #263238;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4fc3f7;
    }
    
    /* Metric box */
    .metric-box {
        background-color: #37474f;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #546e7a;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #f0f0f0;
    }
    
    /* Links */
    a {
        color: #4fc3f7;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #0d47a1;
        color: white;
        border: none;
        border-radius: 0.3rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #263238;
        border-radius: 0.3rem;
        color: #b0bec5;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0d47a1;
        color: white;
    }
    
    /* Success box */
    .success-box {
        background-color: #1b5e20;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4caf50;
    }
    
    /* Highlight box */
    .highlight-box {
        background-color: #263238;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>🎥 Smart Highlight Reel Generator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #616161;'>Create Amazing Badminton Highlights</h3>", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class='info-box'>
    <h4>📹 How It Works</h4>
    <p>Upload your badminton session video, and our AI will automatically detect the most exciting moments, 
    create slow-motion effects, and generate a professional highlight reel for you to share with friends and coaches.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with options
with st.sidebar:
    st.markdown("## 🛠️ Options")
    
    # Highlight detection settings
    st.markdown("### 🎯 Highlight Detection")
    num_highlights = st.slider(
        "Number of Highlights",
        min_value=1,
        max_value=10,
        value=3,
        help="How many highlights to include in your reel"
    )
    
    highlight_duration = st.slider(
        "Highlight Duration (seconds)",
        min_value=2,
        max_value=10,
        value=5,
        help="How long each highlight should be"
    )
    
    # Slow-motion settings
    st.markdown("### ⏱️ Slow Motion")
    slow_motion_factor = st.slider(
        "Slow Motion Factor",
        min_value=0.25,
        max_value=1.0,
        value=0.5,
        step=0.25,
        help="How much to slow down the highlights (0.25 = 4x slower)"
    )
    
    # Transition effects
    st.markdown("### 🎬 Transitions")
    transition_effect = st.selectbox(
        "Transition Effect",
        ["Fade", "Dissolve", "None"],
        help="How to transition between highlights"
    )
    
    # Music options
    st.markdown("### 🎵 Background Music")
    add_music = st.checkbox("Add Background Music", value=True)
    
    if add_music:
        # Add music file selection
        music_file = st.file_uploader(
            "Upload background music (MP3)",
            type=["mp3"],
            help="Upload an MP3 file to use as background music"
        )
        
        if music_file is None:
            st.info("ℹ️ Please upload an MP3 file to add background music to your highlight reel.")
    
    # Export options
    st.markdown("### 📥 Export Options")
    video_quality = st.selectbox(
        "Video Quality",
        ["High (1080p)", "Medium (720p)", "Low (480p)"],
        help="Higher quality = larger file size"
    )
    
    # Tips
    st.markdown("## 💡 Tips")
    st.markdown("""
    - **Best Results**: Use videos with good lighting and clear shots
    - **Processing Time**: Longer videos take more time to process
    - **File Size**: Higher quality settings result in larger files
    - **Sharing**: Download your reel to share on social media
    """)

# Simulate a highlight detection system (stub)
def detect_highlights(video_path, num_highlights=3, highlight_duration=5):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    # Simulate highlight detection (use ML models in real case)
    highlights = []
    for _ in range(num_highlights):
        # Ensure highlights don't overlap and are within video bounds
        if len(highlights) > 0:
            # Find a gap between existing highlights
            valid_starts = []
            for i in range(len(highlights)):
                if i == 0:
                    if highlights[i][0] > highlight_duration:
                        valid_starts.append((0, highlights[i][0] - highlight_duration))
                else:
                    if highlights[i][0] - highlights[i-1][1] > highlight_duration:
                        valid_starts.append((highlights[i-1][1], highlights[i][0] - highlight_duration))
            
            if len(valid_starts) > 0:
                # Choose a random valid start range
                start_range = random.choice(valid_starts)
                start = random.uniform(start_range[0], start_range[1])
            else:
                # If no gaps, try to find a spot at the end
                if duration - highlights[-1][1] > highlight_duration:
                    start = random.uniform(highlights[-1][1], duration - highlight_duration)
                else:
                    # If no space left, skip this highlight
                    continue
        else:
            # First highlight
            start = random.uniform(5, duration - highlight_duration)
        
        end = min(start + highlight_duration, duration)
        
        # Create subclip with slow motion
        subclip = clip.subclip(start, end)
        subclip = subclip.fx(vfx.speedx, 0.5)  # slow-mo
        
        highlights.append((start, end, subclip))
    
    return highlights, clip

# Main content area
st.markdown("<h3 class='sub-header'>📤 Upload Your Video</h3>", unsafe_allow_html=True)

# Upload video
uploaded_video = st.file_uploader("Upload your badminton session video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}")
    tfile.write(uploaded_video.read())
    tfile.close()
    
    # Display original video
    st.markdown("<h3 class='sub-header'>📹 Original Video</h3>", unsafe_allow_html=True)
    st.video(tfile.name)
    
    # Process video
    with st.spinner("Analyzing video and generating highlight reel..."):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress (simulated)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("Analyzing video content...")
            elif i < 60:
                status_text.text("Detecting exciting moments...")
            elif i < 90:
                status_text.text("Creating highlight reel...")
            else:
                status_text.text("Finalizing your highlights...")
        
        # Detect highlights
        highlights, full_clip = detect_highlights(tfile.name, num_highlights, highlight_duration)
        
        # Apply slow motion factor
        for i, (start, end, subclip) in enumerate(highlights):
            highlights[i] = (start, end, subclip.fx(vfx.speedx, slow_motion_factor))
    
    # Success message
    st.markdown("""
    <div class='success-box'>
        <h4>✅ Highlight reel ready!</h4>
        <p>Your highlights have been generated successfully. Scroll down to view and download your reel.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create final clip
    top_clips = [h[2] for h in highlights]
    final_clip = concatenate_videoclips(top_clips)
    
    # Add music if selected and uploaded
    if add_music and music_file is not None:
        try:
            # Save uploaded music file
            music_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            music_path.write(music_file.read())
            music_path.close()
            
            # Load audio clip
            audio_clip = AudioFileClip(music_path.name)
            
            # Create a new audio clip that matches the video duration
            if audio_clip.duration < final_clip.duration:
                # If audio is shorter, create silence for the remaining duration
                from moviepy.audio.AudioClip import AudioClip
                silence = AudioClip(lambda t: 0, duration=final_clip.duration - audio_clip.duration)
                # Concatenate audio with silence
                audio_clip = concatenate_audioclips([audio_clip, silence])
            else:
                # If audio is longer, trim it
                audio_clip = audio_clip.subclip(0, final_clip.duration)
            
            # Set audio volume to 30% of original
            audio_clip = audio_clip.volumex(0.3)
            
            # Set the audio of the final clip
            final_clip = final_clip.set_audio(audio_clip)
            
            # Clean up music file
            os.unlink(music_path.name)
        except Exception as e:
            st.warning(f"Could not add music to the video: {e}")
    
    # Save final output
    output_path = os.path.join(tempfile.gettempdir(), f"highlight_reel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Display final highlight reel
    st.markdown("<h3 class='sub-header'>🎬 Your Highlight Reel</h3>", unsafe_allow_html=True)
    st.video(output_path)
    
    # Display highlight details
    st.markdown("<h3 class='sub-header'>🏸 Highlight Details</h3>", unsafe_allow_html=True)
    
    for idx, (start, end, _) in enumerate(highlights):
        st.markdown(f"""
        <div class='highlight-box'>
            <h4>Highlight {idx + 1}</h4>
            <p>Time: {start:.1f}s to {end:.1f}s (Duration: {end-start:.1f}s)</p>
            <p>Slow Motion: {1/slow_motion_factor:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Download link
    with open(output_path, "rb") as f:
        video_bytes = f.read()
        st.download_button(
            label="📥 Download Highlight Reel",
            data=video_bytes,
            file_name="highlight_reel.mp4",
            mime="video/mp4"
        )
    
    # Clean up
    os.unlink(tfile.name)
    os.unlink(output_path)
else:
    # Instructions when no video is uploaded
    st.markdown("""
    <div class='info-box'>
        <h4>📝 Instructions</h4>
        <ol>
            <li>Upload your badminton session video (MP4, MOV, or AVI format)</li>
            <li>Adjust settings in the sidebar if needed</li>
            <li>Wait for the AI to analyze your video and generate highlights</li>
            <li>Download your highlight reel to share with friends and coaches</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Example video
    st.markdown("<h3 class='sub-header'>📺 Example</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        <p>Here's an example of what your highlight reel might look like:</p>
        <video width='640' height='360' controls>
            <source src='https://www.w3schools.com/html/mov_bbb.mp4' type='video/mp4'>
            Your browser does not support the video tag.
        </video>
    </div>
    """, unsafe_allow_html=True)

# Add a button to start a new session
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🎮 Start New Training Session", type="primary"):
    st.switch_page("pages/1_Game_Mode_AI.py") 