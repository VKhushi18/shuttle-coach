import streamlit as st
import cv2
import numpy as np
from utils.pose_detection import PoseDetector
from utils.object_tracking import ObjectTracker
from utils.movement_analyzer import MovementAnalyzer
from utils.visualization import Visualizer
import tempfile
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Technique Analysis - AI ShuttleCoach",
    page_icon="🏸",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_models(detection_confidence=0.5):
    pose_detector = PoseDetector(min_detection_confidence=detection_confidence)
    object_tracker = ObjectTracker()
    movement_analyzer = MovementAnalyzer()
    visualizer = Visualizer()
    return pose_detector, object_tracker, movement_analyzer, visualizer

# Initialize session state for technique analysis
if 'technique_data' not in st.session_state:
    st.session_state.technique_data = {
        'posture_scores': [],
        'shot_angles': [],
        'footwork_scores': [],
        'shot_types': [],
        'timestamps': [],
        'speed_history': [],
        'trajectory_history': [],
        'detailed_analysis': {
            'posture_breakdown': {},
            'shot_technique_analysis': {},
            'footwork_patterns': {},
            'movement_efficiency': {}
        }
    }

# Title and description
st.title("📊 Technique Analysis")
st.markdown("""
### Detailed analysis of your badminton technique
Upload a video to get comprehensive feedback on your form, movement patterns, and shot execution.
""")

# Sidebar settings
with st.sidebar:
    st.header("🎯 Analysis Settings")
    
    # Analysis focus areas
    analysis_focus = st.multiselect(
        "Select areas to analyze:",
        ["Posture", "Footwork", "Shot Technique", "Court Movement", "Service Analysis"],
        default=["Posture", "Footwork"],
        help="Choose specific aspects of your game to analyze"
    )
    
    # Analysis settings
    frame_rate = st.slider("Frame Analysis Rate", 1, 30, 15, 
                         help="Frames per second to analyze")
    
    detection_confidence = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Adjust the confidence threshold for pose detection"
    )
    
    # Reference pose selection
    st.subheader("🎯 Reference Pose")
    shot_type = st.selectbox(
        "Select shot type for comparison:",
        ["Smash", "Clear", "Drop Shot", "Drive", "Service"]
    )
    
    # Display ideal values for selected shot type
    st.subheader("📐 Ideal Values")
    if shot_type == "Smash":
        st.markdown("""
        **Smash Ideal Values:**
        - **Elbow Angle:** 165° (straight)
        - **Shoulder Height:** 170° (high)
        - **Knee Bend:** 150° (slight)
        - **Racket Head Speed:** 80-100 km/h
        - **Jump Height:** 30-40 cm
        - **Follow-through:** Full extension
        """)
    elif shot_type == "Clear":
        st.markdown("""
        **Clear Ideal Values:**
        - **Elbow Angle:** 155° (bent)
        - **Shoulder Height:** 160° (high)
        - **Knee Bend:** 155° (slight)
        - **Racket Head Speed:** 60-70 km/h
        - **Jump Height:** 20-30 cm
        - **Follow-through:** Full extension
        """)
    elif shot_type == "Drop Shot":
        st.markdown("""
        **Drop Shot Ideal Values:**
        - **Elbow Angle:** 100° (bent)
        - **Shoulder Height:** 130° (moderate)
        - **Knee Bend:** 160° (slight)
        - **Racket Head Speed:** 30-40 km/h
        - **Jump Height:** 10-20 cm
        - **Follow-through:** Gentle
        """)
    elif shot_type == "Drive":
        st.markdown("""
        **Drive Ideal Values:**
        - **Elbow Angle:** 90° (bent)
        - **Shoulder Height:** 90° (moderate)
        - **Knee Bend:** 155° (slight)
        - **Racket Head Speed:** 50-60 km/h
        - **Jump Height:** 5-10 cm
        - **Follow-through:** Quick
        """)
    elif shot_type == "Service":
        st.markdown("""
        **Service Ideal Values:**
        - **Elbow Angle:** 120° (bent)
        - **Shoulder Height:** 100° (moderate)
        - **Knee Bend:** 170° (slight)
        - **Racket Head Speed:** 40-50 km/h
        - **Jump Height:** 0-5 cm
        - **Follow-through:** Gentle
        """)
    
    # Advanced options
    st.subheader("🔍 Advanced Options")
    enable_angle_analysis = st.checkbox("Enable Joint Angle Analysis", True)
    enable_speed_analysis = st.checkbox("Enable Speed Analysis", True)
    enable_trajectory = st.checkbox("Enable Shot Trajectory", True)
    save_analysis = st.checkbox("Save Analysis Report", False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Video Analysis")
    
    # Video upload
    video_file = st.file_uploader(
        "Upload your training video",
        type=['mp4', 'mov', 'avi'],
        help="Supported formats: MP4, MOV, AVI"
    )
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video display container with fixed size
        st.markdown("""
            <style>
            .video-container {
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
            }
            </style>
        """, unsafe_allow_html=True)
        
        video_container = st.empty()
        
        # Create real-time feedback container
        feedback_container = st.empty()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load models
        pose_detector, object_tracker, movement_analyzer, visualizer = load_models(detection_confidence)
        
        # Initialize analysis data collection
        analysis_data = {
            'posture_scores': [],
            'footwork_scores': [],
            'shot_types': [],
            'feedback_history': [],
            'timestamps': [],
            'movement_history': [],
            'joint_angles': [],
            'speed_history': [],
            'current_values': {
                'elbow_angle': 0,
                'shoulder_height': 0,
                'knee_bend': 0,
                'racket_speed': 0,
                'jump_height': 0
            }
        }
        
        # Process video frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            frame_count += 1
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count} of {total_frames}")
            
            # Process frame with pose detector
            results = pose_detector.detect_pose(frame)
            
            # Get player position from pose detection
            player_pos = None
            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[0]
                player_pos = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
                
                # Track movement and calculate speed
                movement_data = movement_analyzer.track_movement(player_pos)
                if len(analysis_data['movement_history']) > 0:
                    last_pos = analysis_data['movement_history'][-1]
                    speed = np.sqrt((player_pos[0] - last_pos[0])**2 + (player_pos[1] - last_pos[1])**2)
                    analysis_data['speed_history'].append(speed)
                    # Convert speed to km/h (approximate)
                    analysis_data['current_values']['racket_speed'] = speed * 3.6
                
                # Store movement data
                analysis_data['movement_history'].append(player_pos)
                
                # Calculate posture score and joint angles
                posture_score = pose_detector.calculate_posture_score(results)
                analysis_data['posture_scores'].append(posture_score)
                
                angles = pose_detector.calculate_joint_angles(results)
                analysis_data['joint_angles'].append(angles)
                
                # Update current values
                if angles:
                    # Use the dominant arm angles (right arm by default)
                    analysis_data['current_values']['elbow_angle'] = angles.get('right_elbow', 0)
                    analysis_data['current_values']['shoulder_height'] = angles.get('right_shoulder', 0)
                    analysis_data['current_values']['knee_bend'] = angles.get('right_knee', 0)
                
                # Calculate jump height (approximate)
                if len(analysis_data['movement_history']) > 10:
                    y_positions = [pos[1] for pos in analysis_data['movement_history'][-10:]]
                    min_y = min(y_positions)
                    max_y = max(y_positions)
                    jump_height = (max_y - min_y) / 10  # Approximate in cm
                    analysis_data['current_values']['jump_height'] = jump_height
                
                # Analyze form and get feedback
                feedback = pose_detector.analyze_form(results)
                if feedback:
                    analysis_data['feedback_history'].append((frame_count, feedback))
                
                # Detect shot type
                shot_type_detected = movement_analyzer.detect_shot_type(results.pose_landmarks.landmark)
                if shot_type_detected:
                    analysis_data['shot_types'].append((frame_count, shot_type_detected))
                
                # Calculate footwork score
                footwork_score = movement_analyzer.calculate_footwork_score(results.pose_landmarks.landmark)
                analysis_data['footwork_scores'].append(footwork_score)
                
                # Store timestamp
                analysis_data['timestamps'].append(time.time())
                
                # Generate real-time feedback based on selected analysis focus
                real_time_feedback = []
                
                # Compare current values with ideal values for selected shot type
                if shot_type == "Smash":
                    if abs(analysis_data['current_values']['elbow_angle'] - 165) > 10:
                        real_time_feedback.append(f"Elbow angle: {analysis_data['current_values']['elbow_angle']:.1f}° (ideal: 165°)")
                    if abs(analysis_data['current_values']['shoulder_height'] - 170) > 10:
                        real_time_feedback.append(f"Shoulder height: {analysis_data['current_values']['shoulder_height']:.1f}° (ideal: 170°)")
                    if abs(analysis_data['current_values']['knee_bend'] - 150) > 10:
                        real_time_feedback.append(f"Knee bend: {analysis_data['current_values']['knee_bend']:.1f}° (ideal: 150°)")
                    if analysis_data['current_values']['racket_speed'] < 60:
                        real_time_feedback.append(f"Racket speed: {analysis_data['current_values']['racket_speed']:.1f} km/h (ideal: 80-100 km/h)")
                    if analysis_data['current_values']['jump_height'] < 20:
                        real_time_feedback.append(f"Jump height: {analysis_data['current_values']['jump_height']:.1f} cm (ideal: 30-40 cm)")
                elif shot_type == "Clear":
                    if abs(analysis_data['current_values']['elbow_angle'] - 155) > 10:
                        real_time_feedback.append(f"Elbow angle: {analysis_data['current_values']['elbow_angle']:.1f}° (ideal: 155°)")
                    if abs(analysis_data['current_values']['shoulder_height'] - 160) > 10:
                        real_time_feedback.append(f"Shoulder height: {analysis_data['current_values']['shoulder_height']:.1f}° (ideal: 160°)")
                    if abs(analysis_data['current_values']['knee_bend'] - 155) > 10:
                        real_time_feedback.append(f"Knee bend: {analysis_data['current_values']['knee_bend']:.1f}° (ideal: 155°)")
                    if analysis_data['current_values']['racket_speed'] < 40:
                        real_time_feedback.append(f"Racket speed: {analysis_data['current_values']['racket_speed']:.1f} km/h (ideal: 60-70 km/h)")
                    if analysis_data['current_values']['jump_height'] < 10:
                        real_time_feedback.append(f"Jump height: {analysis_data['current_values']['jump_height']:.1f} cm (ideal: 20-30 cm)")
                elif shot_type == "Drop Shot":
                    if abs(analysis_data['current_values']['elbow_angle'] - 100) > 10:
                        real_time_feedback.append(f"Elbow angle: {analysis_data['current_values']['elbow_angle']:.1f}° (ideal: 100°)")
                    if abs(analysis_data['current_values']['shoulder_height'] - 130) > 10:
                        real_time_feedback.append(f"Shoulder height: {analysis_data['current_values']['shoulder_height']:.1f}° (ideal: 130°)")
                    if abs(analysis_data['current_values']['knee_bend'] - 160) > 10:
                        real_time_feedback.append(f"Knee bend: {analysis_data['current_values']['knee_bend']:.1f}° (ideal: 160°)")
                    if analysis_data['current_values']['racket_speed'] > 30:
                        real_time_feedback.append(f"Racket speed: {analysis_data['current_values']['racket_speed']:.1f} km/h (ideal: 30-40 km/h)")
                    if analysis_data['current_values']['jump_height'] > 15:
                        real_time_feedback.append(f"Jump height: {analysis_data['current_values']['jump_height']:.1f} cm (ideal: 10-20 cm)")
                elif shot_type == "Drive":
                    if abs(analysis_data['current_values']['elbow_angle'] - 90) > 10:
                        real_time_feedback.append(f"Elbow angle: {analysis_data['current_values']['elbow_angle']:.1f}° (ideal: 90°)")
                    if abs(analysis_data['current_values']['shoulder_height'] - 90) > 10:
                        real_time_feedback.append(f"Shoulder height: {analysis_data['current_values']['shoulder_height']:.1f}° (ideal: 90°)")
                    if abs(analysis_data['current_values']['knee_bend'] - 155) > 10:
                        real_time_feedback.append(f"Knee bend: {analysis_data['current_values']['knee_bend']:.1f}° (ideal: 155°)")
                    if analysis_data['current_values']['racket_speed'] < 30:
                        real_time_feedback.append(f"Racket speed: {analysis_data['current_values']['racket_speed']:.1f} km/h (ideal: 50-60 km/h)")
                    if analysis_data['current_values']['jump_height'] > 5:
                        real_time_feedback.append(f"Jump height: {analysis_data['current_values']['jump_height']:.1f} cm (ideal: 5-10 cm)")
                elif shot_type == "Service":
                    if abs(analysis_data['current_values']['elbow_angle'] - 120) > 10:
                        real_time_feedback.append(f"Elbow angle: {analysis_data['current_values']['elbow_angle']:.1f}° (ideal: 120°)")
                    if abs(analysis_data['current_values']['shoulder_height'] - 100) > 10:
                        real_time_feedback.append(f"Shoulder height: {analysis_data['current_values']['shoulder_height']:.1f}° (ideal: 100°)")
                    if abs(analysis_data['current_values']['knee_bend'] - 170) > 10:
                        real_time_feedback.append(f"Knee bend: {analysis_data['current_values']['knee_bend']:.1f}° (ideal: 170°)")
                    if analysis_data['current_values']['racket_speed'] < 20:
                        real_time_feedback.append(f"Racket speed: {analysis_data['current_values']['racket_speed']:.1f} km/h (ideal: 40-50 km/h)")
                    if analysis_data['current_values']['jump_height'] > 3:
                        real_time_feedback.append(f"Jump height: {analysis_data['current_values']['jump_height']:.1f} cm (ideal: 0-5 cm)")
                
                # Add feedback based on selected analysis focus
                if "Posture" in analysis_focus and posture_score < 70:
                    real_time_feedback.append(f"Posture score: {posture_score:.1f}% (needs improvement)")
                
                if "Footwork" in analysis_focus and footwork_score < 70:
                    real_time_feedback.append(f"Footwork score: {footwork_score:.1f}% (needs improvement)")
                
                if "Shot Technique" in analysis_focus:
                    if shot_type_detected:
                        real_time_feedback.append(f"Detected shot: {shot_type_detected}")
                    else:
                        real_time_feedback.append("No shot detected in current frame")
                
                # Add court movement feedback
                if "Court Movement" in analysis_focus:
                    # Calculate court movement score based on movement history
                    if len(analysis_data['movement_history']) > 5:
                        # Calculate average speed
                        recent_speeds = analysis_data['speed_history'][-5:]
                        avg_speed = sum(recent_speeds) / len(recent_speeds) if recent_speeds else 0
                        
                        # Calculate movement efficiency (how direct the path is)
                        if len(analysis_data['movement_history']) > 10:
                            start_pos = analysis_data['movement_history'][-10]
                            end_pos = analysis_data['movement_history'][-1]
                            direct_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                            actual_distance = sum([np.sqrt((analysis_data['movement_history'][i][0] - analysis_data['movement_history'][i-1][0])**2 + 
                                                         (analysis_data['movement_history'][i][1] - analysis_data['movement_history'][i-1][1])**2) 
                                                for i in range(-9, 0)])
                            
                            efficiency = direct_distance / actual_distance if actual_distance > 0 else 0
                            
                            if efficiency < 0.7:
                                real_time_feedback.append("Movement path not efficient - take more direct routes")
                            
                            if avg_speed < 2:
                                real_time_feedback.append("Movement speed is low - increase court coverage")
                            elif avg_speed > 10:
                                real_time_feedback.append("Movement speed is high - focus on control")
                    
                    # Add court position feedback
                    if player_pos:
                        # Define court zones (approximate)
                        court_width = frame.shape[1]
                        court_height = frame.shape[0]
                        
                        # Check if player is in the back court
                        if player_pos[1] > court_height * 0.7:
                            real_time_feedback.append("Position: Back court")
                        # Check if player is in the front court
                        elif player_pos[1] < court_height * 0.3:
                            real_time_feedback.append("Position: Front court")
                        else:
                            real_time_feedback.append("Position: Mid court")
                        
                        # Check if player is on the left side
                        if player_pos[0] < court_width * 0.4:
                            real_time_feedback.append("Side: Left")
                        # Check if player is on the right side
                        elif player_pos[0] > court_width * 0.6:
                            real_time_feedback.append("Side: Right")
                        else:
                            real_time_feedback.append("Side: Center")
                
                # Add service analysis feedback
                if "Service Analysis" in analysis_focus:
                    # Check for service-specific movements
                    if shot_type_detected == "Service":
                        # Check if the service motion is detected
                        if angles and 'elbow' in angles:
                            if angles['elbow'] < 100:
                                real_time_feedback.append("Service: Elbow too bent")
                            elif angles['elbow'] > 140:
                                real_time_feedback.append("Service: Elbow too straight")
                        
                        # Check for proper service stance
                        if player_pos:
                            # Check if player is in the correct service position
                            if player_pos[1] > frame.shape[0] * 0.8:
                                real_time_feedback.append("Service: Too far back in court")
                            elif player_pos[1] < frame.shape[0] * 0.2:
                                real_time_feedback.append("Service: Too close to net")
                    
                    # Add general service feedback if no specific service is detected
                    elif not shot_type_detected:
                        real_time_feedback.append("Service: No service motion detected")
                
                # Display real-time feedback
                if real_time_feedback:
                    feedback_container.markdown("""
                    <div style="background-color: rgba(0, 0, 0, 0.7); padding: 10px; border-radius: 5px; color: white;">
                        <h4 style="margin-top: 0;">Real-time Feedback</h4>
                        {}
                    </div>
                    """.format("<br>".join([f"• {f}" for f in real_time_feedback])), unsafe_allow_html=True)
            
            # Draw pose landmarks with advanced visualization
            if results.pose_landmarks:
                # Draw basic pose
                frame = pose_detector.draw_pose(frame, results)
                
                # Draw advanced pose with angles
                angles = pose_detector.calculate_joint_angles(results)
                frame = pose_detector.draw_advanced_pose(frame, results, show_angles=True)
                
                # Remove court lines overlay
                # frame = visualizer.draw_court(frame)
                
                # Draw player position and movement trail
                if player_pos:
                    cv2.circle(frame, player_pos, 10, (0, 255, 0), 2)
                    # Draw movement trail
                    if len(analysis_data['movement_history']) > 1:
                        for i in range(1, min(10, len(analysis_data['movement_history']))):
                            cv2.line(frame, 
                                    analysis_data['movement_history'][-i-1],
                                    analysis_data['movement_history'][-i],
                                    (0, 255, 0), 2)
                
                # Overlay feedback directly on the video
                if real_time_feedback:
                    # Create a semi-transparent overlay for text with gradient background
                    overlay = frame.copy()
                    # Create a gradient background for the overlay
                    for i in range(150):
                        alpha = 0.8 - (i / 150) * 0.3  # Gradient alpha from 0.8 to 0.5
                        cv2.line(overlay, (10, i+10), (630, i+10), (0, 0, 0), 1)
                        # Fix: Don't try to set alpha channel directly
                        for j in range(620):
                            # Just set the BGR values without alpha
                            overlay[i+10, j+10] = (0, 0, 0)
                    
                    # Apply the overlay with transparency
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Add feedback text with better font and colors
                    # Title with gradient color
                    cv2.putText(frame, "REAL-TIME FEEDBACK", (20, 40), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Add a separator line
                    cv2.line(frame, (20, 50), (620, 50), (255, 255, 255), 1)
                    
                    # Group feedback by category
                    technique_feedback = []
                    footwork_feedback = []
                    general_feedback = []
                    
                    for feedback in real_time_feedback:
                        if "angle" in feedback.lower() or "speed" in feedback.lower() or "height" in feedback.lower():
                            technique_feedback.append(feedback)
                        elif "footwork" in feedback.lower() or "movement" in feedback.lower():
                            footwork_feedback.append(feedback)
                        else:
                            general_feedback.append(feedback)
                    
                    # Display technique feedback with blue color
                    y_pos = 70
                    if technique_feedback:
                        cv2.putText(frame, "TECHNIQUE:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 191, 255), 2)
                        y_pos += 25
                        for feedback in technique_feedback:
                            cv2.putText(frame, feedback, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 191, 255), 1)
                            y_pos += 20
                    
                    # Display footwork feedback with green color
                    if footwork_feedback:
                        cv2.putText(frame, "FOOTWORK:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_pos += 25
                        for feedback in footwork_feedback:
                            cv2.putText(frame, feedback, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            y_pos += 20
                    
                    # Display general feedback with white color
                    if general_feedback:
                        cv2.putText(frame, "GENERAL:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_pos += 25
                        for feedback in general_feedback:
                            cv2.putText(frame, feedback, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_pos += 20
                    
                    # Add foot movement suggestions based on shot type
                    if shot_type == "Smash" and "Footwork" in analysis_focus:
                        # Add footwork suggestions for smash
                        footwork_suggestions = [
                            "Step forward with left foot",
                            "Push off with right foot",
                            "Land with feet shoulder-width apart",
                            "Recover to ready position quickly"
                        ]
                        cv2.putText(frame, "FOOTWORK SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        y_pos += 25
                        for suggestion in footwork_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                            y_pos += 20
                    elif shot_type == "Clear" and "Footwork" in analysis_focus:
                        # Add footwork suggestions for clear
                        footwork_suggestions = [
                            "Step back with right foot",
                            "Transfer weight to back foot",
                            "Step forward with left foot",
                            "Follow through with right foot"
                        ]
                        cv2.putText(frame, "FOOTWORK SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        y_pos += 25
                        for suggestion in footwork_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                            y_pos += 20
                    elif shot_type == "Drop Shot" and "Footwork" in analysis_focus:
                        # Add footwork suggestions for drop shot
                        footwork_suggestions = [
                            "Step forward with left foot",
                            "Keep weight on front foot",
                            "Minimal jump, focus on control",
                            "Quick recovery to center"
                        ]
                        cv2.putText(frame, "FOOTWORK SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        y_pos += 25
                        for suggestion in footwork_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                            y_pos += 20
                    elif shot_type == "Drive" and "Footwork" in analysis_focus:
                        # Add footwork suggestions for drive
                        footwork_suggestions = [
                            "Quick shuffle steps",
                            "Stay low with bent knees",
                            "Keep weight centered",
                            "Explosive first step"
                        ]
                        cv2.putText(frame, "FOOTWORK SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        y_pos += 25
                        for suggestion in footwork_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                            y_pos += 20
                    elif shot_type == "Service" and "Footwork" in analysis_focus:
                        # Add footwork suggestions for service
                        footwork_suggestions = [
                            "Feet shoulder-width apart",
                            "Left foot slightly forward",
                            "Weight on back foot",
                            "Step forward with serve"
                        ]
                        cv2.putText(frame, "FOOTWORK SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                        y_pos += 25
                        for suggestion in footwork_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                            y_pos += 20
                    
                    # Add relevant suggestions based on analysis focus
                    if "Posture" in analysis_focus and posture_score < 70:
                        # Add posture suggestions
                        posture_suggestions = [
                            "Keep back straight",
                            "Shoulders level",
                            "Head up looking forward",
                            "Bend knees slightly"
                        ]
                        cv2.putText(frame, "POSTURE SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        y_pos += 25
                        for suggestion in posture_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                            y_pos += 20
                    
                    if "Shot Technique" in analysis_focus and shot_type_detected:
                        # Add shot technique suggestions
                        if shot_type_detected == "Smash":
                            technique_suggestions = [
                                "Extend arm fully at contact",
                                "Snap wrist at impact",
                                "Follow through downward",
                                "Keep elbow high"
                            ]
                        elif shot_type_detected == "Clear":
                            technique_suggestions = [
                                "High contact point",
                                "Full arm extension",
                                "Follow through upward",
                                "Transfer weight forward"
                            ]
                        elif shot_type_detected == "Drop Shot":
                            technique_suggestions = [
                                "Soft grip for control",
                                "Gentle wrist action",
                                "High contact point",
                                "Minimal follow-through"
                            ]
                        elif shot_type_detected == "Drive":
                            technique_suggestions = [
                                "Quick wrist snap",
                                "Flat racket face",
                                "Contact at net height",
                                "Short, sharp follow-through"
                            ]
                        else:
                            technique_suggestions = [
                                "Consistent contact point",
                                "Proper grip pressure",
                                "Appropriate follow-through",
                                "Balance throughout shot"
                            ]
                        
                        cv2.putText(frame, "TECHNIQUE SUGGESTIONS:", (20, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
                        y_pos += 25
                        for suggestion in technique_suggestions:
                            cv2.putText(frame, suggestion, (30, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1)
                            y_pos += 20

            # Resize frame for display
            display_width = 640
            display_height = int(height * (display_width / width))
            frame = cv2.resize(frame, (display_width, display_height))
            
            # Display frame
            video_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Remove the separate feedback container since we're overlaying it on the video
            # feedback_container.empty()
            
            # Add a small delay to control playback speed
            time.sleep(1/fps)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        # Store analysis data in session state
        st.session_state.technique_data = analysis_data

with col2:
    st.subheader("Analysis Results")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Posture Analysis", "Shot Analysis", "Movement Analysis"])
    
    with tab1:
        st.write("Posture analysis results will appear here")
        
    with tab2:
        st.write("Shot analysis results will appear here")
        
    with tab3:
        st.write("Movement analysis results will appear here")

# Cleanup
if 'video_path' in locals():
    try:
        os.unlink(video_path)
    except Exception as e:
        st.warning(f"Could not delete temporary file: {e}") 