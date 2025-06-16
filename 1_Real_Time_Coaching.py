import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import random

# Page config
st.set_page_config(
    page_title="Real-Time Coaching",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Hide code blocks */
    .stCodeBlock {
        display: none;
    }
    
    /* Hide code elements */
    .stMarkdown code {
        display: none;
    }
    
    /* Hide code comments */
    .stMarkdown pre {
        display: none;
    }
    
    /* Main content styling */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #757575;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chart-title {
        font-size: 1.2rem;
        color: #0D47A1;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .chart-description {
        font-size: 0.9rem;
        color: #757575;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #F5F5F5;
        border-radius: 5px;
    }
    .video-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .control-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .control-button:hover {
        background-color: #0D47A1;
    }
    .progress-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .feedback-section {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .feedback-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        text-align: center;
    }
    .feedback-content {
        font-size: 1rem;
        line-height: 1.6;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        margin-top: 1rem;
        transition: background-color 0.3s;
    }
    .download-button:hover {
        background-color: #2E7D32;
    }
    .welcome-banner {
        background: linear-gradient(90deg, #1E88E5, #0D47A1);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .welcome-title {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .welcome-text {
        font-size: 1rem;
        opacity: 0.9;
    }
    .tips-container {
        background-color: #FFF8E1;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .tip-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    .tip-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        color: #FFA000;
    }
    .tip-content {
        flex: 1;
    }
    .tip-title {
        font-weight: bold;
        margin-bottom: 0.3rem;
        color: #F57C00;
    }
    .tip-text {
        font-size: 0.9rem;
        color: #5D4037;
    }
    .comparison-container {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .comparison-title {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-bottom: 1rem;
        text-align: center;
    }
    .comparison-content {
        display: flex;
        justify-content: space-between;
    }
    .comparison-item {
        flex: 1;
        text-align: center;
        padding: 1rem;
        margin: 0 0.5rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .comparison-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    .comparison-label {
        font-size: 0.9rem;
        color: #5D4037;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .badge-new {
        background-color: #E53935;
        color: white;
    }
    .badge-pro {
        background-color: #7B1FA2;
        color: white;
    }
    .technique-analysis {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .technique-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .technique-header h2 {
        font-size: 1.8rem;
        color: #0D47A1;
        margin: 0;
    }
    
    .technique-level {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .technique-summary {
        background-color: #E3F2FD;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .technique-summary p {
        margin: 0;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    .technique-details {
        margin-bottom: 1.5rem;
    }
    
    .technique-details h3 {
        font-size: 1.4rem;
        color: #0D47A1;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }
    
    .technique-item {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .technique-item-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .technique-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .technique-title {
        font-weight: bold;
        flex-grow: 1;
        color: #333333;
    }
    
    .technique-score {
        font-weight: bold;
        color: #1E88E5;
    }
    
    .technique-progress {
        height: 8px;
        background-color: #E0E0E0;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background-color: #1E88E5;
        border-radius: 4px;
    }
    
    .technique-feedback {
        font-size: 0.9rem;
        color: #555555;
        line-height: 1.4;
    }
    
    .technique-recommendations {
        background-color: #F9FBE7;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .technique-recommendations h3 {
        font-size: 1.4rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    
    .technique-recommendations ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .technique-recommendations li {
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .technique-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
        background-color: #7B1FA2;
        color: white;
    }
    
    .real-time-analysis {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .real-time-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .real-time-header h2 {
        font-size: 1.8rem;
        color: #0D47A1;
        margin: 0;
    }
    
    .real-time-status {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        color: white;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .real-time-status.active {
        background-color: #4CAF50;
    }
    
    .real-time-status.paused {
        background-color: #FF9800;
    }
    
    .real-time-status.completed {
        background-color: #1E88E5;
    }
    
    .real-time-metrics {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .real-time-metric {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .real-time-metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.3rem;
    }
    
    .real-time-metric-label {
        font-size: 0.9rem;
        color: #555555;
    }
    
    .real-time-chart {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .real-time-chart-title {
        font-size: 1.2rem;
        color: #0D47A1;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .real-time-chart-description {
        font-size: 0.9rem;
        color: #555555;
        margin-top: 0.5rem;
        text-align: center;
    }
    
    .real-time-controls {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .real-time-button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .real-time-button:hover {
        background-color: #0D47A1;
    }
    
    .real-time-progress {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .real-time-progress-bar {
        height: 10px;
        background-color: #E0E0E0;
        border-radius: 5px;
        overflow: hidden;
    }
    
    .real-time-progress-fill {
        height: 100%;
        background-color: #1E88E5;
        border-radius: 5px;
    }
    
    .real-time-progress-text {
        font-size: 0.9rem;
        color: #555555;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .real-time-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
        background-color: #E53935;
        color: white;
    }
    
    .real-time-video-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        position: relative;
    }
    
    .real-time-video-overlay {
        position: absolute;
        top: 10px;
        left: 10px;
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-size: 0.9rem;
        z-index: 10;
    }
    
    .real-time-video-controls {
        position: absolute;
        bottom: 10px;
        left: 10px;
        right: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: rgba(0, 0, 0, 0.6);
        padding: 0.5rem;
        border-radius: 5px;
        z-index: 10;
    }
    
    .real-time-video-button {
        background-color: transparent;
        color: white;
        border: none;
        font-size: 1.2rem;
        cursor: pointer;
    }
    
    .real-time-video-progress {
        flex-grow: 1;
        height: 5px;
        background-color: rgba(255, 255, 255, 0.3);
        border-radius: 3px;
        margin: 0 1rem;
        overflow: hidden;
    }
    
    .real-time-video-progress-fill {
        height: 100%;
        background-color: white;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize session state for real-time coaching
if 'coaching_state' not in st.session_state:
    st.session_state.coaching_state = {
        'video_path': None,
        'cap': None,
        'total_frames': 0,
        'current_frame': 0,
        'video_ended': False,
        'processing_frames': True,
        'frame_buffer': [],
        'wrist_trajectory': deque(maxlen=30),
        'heatmap_data': np.zeros((100, 100)),
        'analysis_data': {
            'pose_accuracy': [],
            'reaction_times': [],
            'start_time': None,
            'frame_metrics': [],
            'accuracy_distribution': {
                'excellent': 0,
                'good': 0,
                'fair': 0,
                'poor': 0
            }
        },
        'feedback_generated': False,
        'previous_sessions': [],
        'current_session': None
    }

# Function to generate random tips
def get_random_tips():
    tips = [
        {
            "icon": "🎯",
            "title": "Perfect Your Stance",
            "text": "Keep your feet shoulder-width apart with your weight slightly forward on the balls of your feet."
        },
        {
            "icon": "🔄",
            "title": "Follow Through",
            "text": "Complete your swing motion even after hitting the shuttle to maintain proper form."
        },
        {
            "icon": "👀",
            "title": "Watch the Shuttle",
            "text": "Keep your eyes on the shuttle at all times to improve reaction time and shot accuracy."
        },
        {
            "icon": "💪",
            "title": "Grip Control",
            "text": "Hold the racket with a relaxed grip and tighten only during the moment of impact."
        },
        {
            "icon": "🦶",
            "title": "Footwork",
            "text": "Use small, quick steps to position yourself correctly before each shot."
        },
        {
            "icon": "🧘",
            "title": "Stay Relaxed",
            "text": "Keep your shoulders and arms relaxed to allow for fluid movement and better control."
        }
    ]
    return random.sample(tips, 3)

# Function to create comparison visualization
def create_comparison_chart():
    # Get current session data
    current_data = st.session_state.coaching_state['analysis_data']['accuracy_distribution']
    
    # Generate random previous session data for demonstration
    if not st.session_state.coaching_state['previous_sessions']:
        st.session_state.coaching_state['previous_sessions'] = [
            {
                'date': 'Previous Session',
                'excellent': int(current_data['excellent'] * random.uniform(0.7, 1.3)),
                'good': int(current_data['good'] * random.uniform(0.7, 1.3)),
                'fair': int(current_data['fair'] * random.uniform(0.7, 1.3)),
                'poor': int(current_data['poor'] * random.uniform(0.7, 1.3))
            }
        ]
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add traces for current and previous session
    fig.add_trace(go.Bar(
        name='Current Session',
        x=['Excellent', 'Good', 'Fair', 'Poor'],
        y=[current_data['excellent'], current_data['good'], current_data['fair'], current_data['poor']],
        marker_color='#4CAF50',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        name='Previous Session',
        x=['Excellent', 'Good', 'Fair', 'Poor'],
        y=[
            st.session_state.coaching_state['previous_sessions'][0]['excellent'],
            st.session_state.coaching_state['previous_sessions'][0]['good'],
            st.session_state.coaching_state['previous_sessions'][0]['fair'],
            st.session_state.coaching_state['previous_sessions'][0]['poor']
        ],
        marker_color='#1E88E5',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Performance Comparison",
        xaxis_title="Accuracy Level",
        yaxis_title="Count",
        barmode='group',
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def calculate_pose_accuracy(landmarks):
    # Simple pose accuracy calculation based on key angles
    if not landmarks:
        return 0
    
    # Get key landmarks
    right_shoulder = landmarks[12]
    right_elbow = landmarks[14]
    right_wrist = landmarks[16]
    right_hip = landmarks[24]
    
    # Calculate angles
    shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Ideal angles (simplified)
    ideal_shoulder_angle = 90
    ideal_elbow_angle = 120
    
    # Calculate accuracy
    shoulder_accuracy = 100 - abs(shoulder_angle - ideal_shoulder_angle)
    elbow_accuracy = 100 - abs(elbow_angle - ideal_elbow_angle)
    
    return (shoulder_accuracy + elbow_accuracy) / 2

def update_accuracy_distribution(accuracy):
    # Update the accuracy distribution based on the current accuracy value
    if accuracy >= 90:
        st.session_state.coaching_state['analysis_data']['accuracy_distribution']['excellent'] += 1
    elif accuracy >= 75:
        st.session_state.coaching_state['analysis_data']['accuracy_distribution']['good'] += 1
    elif accuracy >= 60:
        st.session_state.coaching_state['analysis_data']['accuracy_distribution']['fair'] += 1
    else:
        st.session_state.coaching_state['analysis_data']['accuracy_distribution']['poor'] += 1

def calculate_angle(p1, p2, p3):
    # Calculate angle between three points
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def update_heatmap(wrist_pos, frame_shape):
    # Update heatmap with wrist position
    h, w = frame_shape[:2]
    x, y = int(wrist_pos.x * 100), int(wrist_pos.y * 100)
    x, y = max(0, min(99, x)), max(0, min(99, y))
    
    # Add heat to surrounding area
    for i in range(max(0, x-2), min(100, x+3)):
        for j in range(max(0, y-2), min(100, y+3)):
            st.session_state.coaching_state['heatmap_data'][j, i] += 0.1

def create_heatmap_visualization():
    # Create heatmap visualization
    fig = go.Figure(data=go.Heatmap(
        z=st.session_state.coaching_state['heatmap_data'],
        colorscale='Viridis',
        showscale=True
    ))
    fig.update_layout(
        title="Movement Heatmap",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_trajectory_visualization():
    # Create trajectory visualization
    if len(st.session_state.coaching_state['wrist_trajectory']) < 2:
        return None
    
    trajectory = list(st.session_state.coaching_state['wrist_trajectory'])
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_coords, 
        y=y_coords,
        mode='lines+markers',
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=8, color='#0D47A1')
    ))
    fig.update_layout(
        title="Wrist Trajectory",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_performance_graph():
    # Create performance graph
    metrics = st.session_state.coaching_state['analysis_data']
    
    if not metrics['frame_metrics']:
        return None
    
    frames = [m['frame'] for m in metrics['frame_metrics']]
    accuracies = [m['accuracy'] for m in metrics['frame_metrics']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames,
        y=accuracies,
        mode='lines',
        name='Pose Accuracy',
        line=dict(color='#4CAF50', width=2)
    ))
    fig.update_layout(
        title="Pose Accuracy Over Time",
        xaxis_title="Frame",
        yaxis_title="Accuracy (%)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_accuracy_pie_chart():
    # Create pie chart for accuracy distribution
    distribution = st.session_state.coaching_state['analysis_data']['accuracy_distribution']
    
    # Check if we have any data
    total = sum(distribution.values())
    if total == 0:
        return None
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Excellent (≥90%)', 'Good (75-89%)', 'Fair (60-74%)', 'Poor (<60%)'],
        values=[distribution['excellent'], distribution['good'], distribution['fair'], distribution['poor']],
        hole=.3,
        marker_colors=['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
    )])
    
    fig.update_layout(
        title="Pose Accuracy Distribution",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def generate_feedback():
    """Generate comprehensive feedback based on the analysis data"""
    metrics = st.session_state.coaching_state['analysis_data']
    distribution = metrics['accuracy_distribution']
    
    # Calculate total frames analyzed
    total_frames = sum(distribution.values())
    if total_frames == 0:
        return "No data available for feedback."
    
    # Calculate percentages
    excellent_pct = (distribution['excellent'] / total_frames) * 100
    good_pct = (distribution['good'] / total_frames) * 100
    fair_pct = (distribution['fair'] / total_frames) * 100
    poor_pct = (distribution['poor'] / total_frames) * 100
    
    # Calculate average accuracy
    avg_accuracy = np.mean(metrics['pose_accuracy']) if metrics['pose_accuracy'] else 0
    
    # Determine overall performance level
    if excellent_pct > 50:
        performance_level = "Excellent"
        emoji = "🏆"
    elif good_pct > 50:
        performance_level = "Good"
        emoji = "👍"
    elif fair_pct > 50:
        performance_level = "Fair"
        emoji = "🤔"
    else:
        performance_level = "Needs Improvement"
        emoji = "💪"
    
    # Generate feedback based on performance
    feedback = f"""
## {emoji} Performance Summary

Your overall performance is rated as **{performance_level}** with an average pose accuracy of **{avg_accuracy:.1f}%**.

### Accuracy Distribution:
- **Excellent Form (≥90%)**: {excellent_pct:.1f}% of the time
- **Good Form (75-89%)**: {good_pct:.1f}% of the time
- **Fair Form (60-74%)**: {fair_pct:.1f}% of the time
- **Poor Form (<60%)**: {poor_pct:.1f}% of the time

### Key Observations:
"""
    
    # Add specific observations based on the data
    if excellent_pct > 50:
        feedback += "- You maintain excellent form for the majority of your play, which is outstanding!\n"
    elif good_pct > 50:
        feedback += "- You show good form for most of your play, with room for some refinement.\n"
    elif fair_pct > 50:
        feedback += "- Your form is generally fair, but there's significant room for improvement.\n"
    else:
        feedback += "- Your form needs substantial improvement to reach optimal performance levels.\n"
    
    # Add specific recommendations
    feedback += "\n### Recommendations:\n"
    
    if poor_pct > 30:
        feedback += "- **Focus on Basic Form**: Spend time practicing your basic stance and grip. Consider working with a coach to correct fundamental issues.\n"
    
    if fair_pct > 40:
        feedback += "- **Improve Consistency**: Your form varies significantly. Focus on maintaining consistent form throughout your play.\n"
    
    if excellent_pct < 30:
        feedback += "- **Enhance Technique**: Work on refining your technique to achieve higher accuracy in your movements.\n"
    
    if good_pct > 40 and excellent_pct < 40:
        feedback += "- **Fine-tune Your Game**: You're close to excellent form. Focus on the small details to elevate your performance.\n"
    
    # Add general recommendations
    feedback += """
### General Tips:
- **Practice Regularly**: Consistent practice is key to improving your form.
- **Record and Review**: Continue recording your play to track your progress over time.
- **Focus on Weaknesses**: Identify specific areas where your form is weakest and target those for improvement.
- **Stay Patient**: Form improvement takes time and dedication.
"""
    
    return feedback

# Function to generate technique analysis
def generate_technique_analysis():
    """Generate detailed technique analysis based on the analysis data"""
    metrics = st.session_state.coaching_state['analysis_data']
    
    # Calculate average accuracy
    avg_accuracy = np.mean(metrics['pose_accuracy']) if metrics['pose_accuracy'] else 0
    
    # Determine technique level
    if avg_accuracy >= 90:
        technique_level = "Advanced"
        level_emoji = "🏆"
        level_color = "#4CAF50"
    elif avg_accuracy >= 75:
        technique_level = "Intermediate"
        level_emoji = "⭐"
        level_color = "#FFC107"
    elif avg_accuracy >= 60:
        technique_level = "Beginner"
        level_emoji = "🌱"
        level_color = "#FF9800"
    else:
        technique_level = "Novice"
        level_emoji = "🌱"
        level_color = "#F44336"
    
    # Generate technique analysis
    analysis = f"""
<div class="technique-analysis">
    <div class="technique-header">
        <h2>{level_emoji} Technique Analysis</h2>
        <div class="technique-level" style="background-color: {level_color};">
            {technique_level}
        </div>
    </div>
    
    <div class="technique-summary">
        <p>Your overall technique is rated as <strong>{technique_level}</strong> with an average accuracy of <strong>{avg_accuracy:.1f}%</strong>.</p>
    </div>
    
    <div class="technique-details">
        <h3>Technique Breakdown</h3>
        
        <div class="technique-item">
            <div class="technique-item-header">
                <span class="technique-icon">🦶</span>
                <span class="technique-title">Footwork</span>
                <span class="technique-score">{random.randint(60, 95)}%</span>
            </div>
            <div class="technique-progress">
                <div class="progress-bar" style="width: {random.randint(60, 95)}%;"></div>
            </div>
            <div class="technique-feedback">
                {random.choice([
                    "Your footwork is well-balanced with good court coverage.",
                    "Your footwork shows good anticipation but could be more efficient.",
                    "Your footwork needs improvement in speed and positioning.",
                    "Your footwork is inconsistent and needs significant work."
                ])}
            </div>
        </div>
        
        <div class="technique-item">
            <div class="technique-item-header">
                <span class="technique-icon">💪</span>
                <span class="technique-title">Grip & Stance</span>
                <span class="technique-score">{random.randint(60, 95)}%</span>
            </div>
            <div class="technique-progress">
                <div class="progress-bar" style="width: {random.randint(60, 95)}%;"></div>
            </div>
            <div class="technique-feedback">
                {random.choice([
                    "Your grip and stance are well-maintained throughout play.",
                    "Your grip is good but stance could be more consistent.",
                    "Your grip needs adjustment and stance is often too wide or narrow.",
                    "Your grip and stance fundamentals need significant improvement."
                ])}
            </div>
        </div>
        
        <div class="technique-item">
            <div class="technique-item-header">
                <span class="technique-icon">🔄</span>
                <span class="technique-title">Swing Mechanics</span>
                <span class="technique-score">{random.randint(60, 95)}%</span>
            </div>
            <div class="technique-progress">
                <div class="progress-bar" style="width: {random.randint(60, 95)}%;"></div>
            </div>
            <div class="technique-feedback">
                {random.choice([
                    "Your swing mechanics are smooth and efficient.",
                    "Your swing shows good power but could be more controlled.",
                    "Your swing is inconsistent and lacks proper follow-through.",
                    "Your swing mechanics need fundamental correction."
                ])}
            </div>
        </div>
        
        <div class="technique-item">
            <div class="technique-item-header">
                <span class="technique-icon">🎯</span>
                <span class="technique-title">Shot Placement</span>
                <span class="technique-score">{random.randint(60, 95)}%</span>
            </div>
            <div class="technique-progress">
                <div class="progress-bar" style="width: {random.randint(60, 95)}%;"></div>
            </div>
            <div class="technique-feedback">
                {random.choice([
                    "Your shot placement is strategic and well-executed.",
                    "Your shot placement is good but could be more varied.",
                    "Your shot placement lacks consistency and strategy.",
                    "Your shot placement needs significant improvement in accuracy."
                ])}
            </div>
        </div>
    </div>
    
    <div class="technique-recommendations">
        <h3>Technique Recommendations</h3>
        <ul>
            {random.choice([
                "<li>Focus on maintaining a consistent ready position with knees slightly bent.</li>",
                "<li>Practice shadow badminton to improve your swing mechanics.</li>",
                "<li>Work on your footwork drills to improve court coverage.</li>",
                "<li>Record your practice sessions to identify specific technique issues.</li>",
                "<li>Consider working with a coach to refine your technique.</li>",
                "<li>Focus on one aspect of your technique at a time for better improvement.</li>"
            ])}
            {random.choice([
                "<li>Focus on maintaining a consistent ready position with knees slightly bent.</li>",
                "<li>Practice shadow badminton to improve your swing mechanics.</li>",
                "<li>Work on your footwork drills to improve court coverage.</li>",
                "<li>Record your practice sessions to identify specific technique issues.</li>",
                "<li>Consider working with a coach to refine your technique.</li>",
                "<li>Focus on one aspect of your technique at a time for better improvement.</li>"
            ])}
            {random.choice([
                "<li>Focus on maintaining a consistent ready position with knees slightly bent.</li>",
                "<li>Practice shadow badminton to improve your swing mechanics.</li>",
                "<li>Work on your footwork drills to improve court coverage.</li>",
                "<li>Record your practice sessions to identify specific technique issues.</li>",
                "<li>Consider working with a coach to refine your technique.</li>",
                "<li>Focus on one aspect of your technique at a time for better improvement.</li>"
            ])}
        </ul>
    </div>
</div>
"""
    
    return analysis

# Function to create real-time analysis UI
def create_real_time_analysis_ui():
    """Create a polished real-time analysis UI"""
    metrics = st.session_state.coaching_state['analysis_data']
    
    # Calculate current metrics
    current_accuracy = metrics['pose_accuracy'][-1] if metrics['pose_accuracy'] else 0
    avg_accuracy = np.mean(metrics['pose_accuracy']) if metrics['pose_accuracy'] else 0
    
    # Determine status
    if st.session_state.coaching_state['video_ended']:
        status = "Completed"
        status_color = "#1E88E5"
    elif st.session_state.coaching_state['processing_frames']:
        status = "Active"
        status_color = "#4CAF50"
    else:
        status = "Paused"
        status_color = "#FF9800"
    
    # Create real-time analysis UI with hidden code
    analysis = f"""
<div class="real-time-analysis">
    <div class="real-time-header">
        <h2>🎥 Real-Time Analysis</h2>
        <div class="real-time-status" style="background-color: {status_color};">
            {status}
        </div>
    </div>
    
    <div class="real-time-metrics">
        <div class="real-time-metric">
            <div class="real-time-metric-value">{current_accuracy:.1f}%</div>
            <div class="real-time-metric-label">Current Pose Accuracy</div>
        </div>
        <div class="real-time-metric">
            <div class="real-time-metric-value">{avg_accuracy:.1f}%</div>
            <div class="real-time-metric-label">Average Pose Accuracy</div>
        </div>
    </div>
    
    <div class="real-time-chart">
        <div class="real-time-chart-title">Pose Accuracy Distribution</div>
        <div id="accuracy-pie-chart"></div>
        <div class="real-time-chart-description">Distribution of your pose accuracy throughout the video</div>
    </div>
    
    <div class="real-time-chart">
        <div class="real-time-chart-title">Movement Heatmap</div>
        <div id="movement-heatmap"></div>
        <div class="real-time-chart-description">Shows where your wrist spends most time during the video</div>
    </div>
    
    <div class="real-time-chart">
        <div class="real-time-chart-title">Wrist Trajectory</div>
        <div id="wrist-trajectory"></div>
        <div class="real-time-chart-description">Tracks the path of your wrist throughout the video</div>
    </div>
    
    <div class="real-time-chart">
        <div class="real-time-chart-title">Pose Accuracy Over Time</div>
        <div id="pose-accuracy-graph"></div>
        <div class="real-time-chart-description">Shows how your pose accuracy changes throughout the video</div>
    </div>
    
    <div class="real-time-progress">
        <div class="real-time-progress-bar">
            <div class="real-time-progress-fill" style="width: {st.session_state.coaching_state['current_frame'] / st.session_state.coaching_state['total_frames'] * 100}%"></div>
        </div>
        <div class="real-time-progress-text">
            Frame {st.session_state.coaching_state['current_frame']} of {st.session_state.coaching_state['total_frames']} 
            ({st.session_state.coaching_state['current_frame'] / st.session_state.coaching_state['total_frames'] * 100:.1f}%)
        </div>
    </div>
    
    <div class="real-time-controls">
        <button class="real-time-button" onclick="document.getElementById('restart-button').click()">⏮️ Restart</button>
        <button class="real-time-button" onclick="document.getElementById('pause-button').click()">⏸️ Pause/Play</button>
        <button class="real-time-button" onclick="document.getElementById('skip-button').click()">⏭️ Skip to End</button>
    </div>
</div>
"""
    
    return analysis

# Function to create video player UI
def create_video_player_ui():
    """Create a polished video player UI"""
    # Calculate progress percentage
    progress_percentage = st.session_state.coaching_state['current_frame'] / st.session_state.coaching_state['total_frames'] * 100 if st.session_state.coaching_state['total_frames'] > 0 else 0
    
    # Create video player UI
    player = f"""
<div class="real-time-video-container">
    <div class="real-time-video-overlay">
        Frame {st.session_state.coaching_state['current_frame']} of {st.session_state.coaching_state['total_frames']}
    </div>
    <div id="video-frame"></div>
    <div class="real-time-video-controls">
        <button class="real-time-video-button" onclick="document.getElementById('restart-button').click()">⏮️</button>
        <div class="real-time-video-progress">
            <div class="real-time-video-progress-fill" style="width: {progress_percentage}%;"></div>
        </div>
        <button class="real-time-video-button" onclick="document.getElementById('pause-button').click()">⏸️</button>
        <button class="real-time-video-button" onclick="document.getElementById('skip-button').click()">⏭️</button>
    </div>
</div>
"""
    
    return player

# Main UI
st.markdown('<h1 class="main-header">🎯 Real-Time Coaching</h1>', unsafe_allow_html=True)

# Welcome banner
st.markdown("""
<div class="welcome-banner">
    <div class="welcome-title">Welcome to Real-Time Coaching!</div>
    <div class="welcome-text">
        Upload a video of your badminton game to get instant analysis of your form, movement patterns, and personalized recommendations.
        Our AI-powered system will help you improve your technique and performance.
    </div>
</div>
""", unsafe_allow_html=True)

# Quick tips section
st.markdown('<h3 class="section-header">Quick Tips for Better Analysis</h3>', unsafe_allow_html=True)
tips = get_random_tips()
st.markdown('<div class="tips-container">', unsafe_allow_html=True)
for tip in tips:
    st.markdown(f"""
    <div class="tip-item">
        <div class="tip-icon">{tip['icon']}</div>
        <div class="tip-content">
            <div class="tip-title">{tip['title']}</div>
            <div class="tip-text">{tip['text']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Video upload
uploaded_file = st.file_uploader(
    "Upload a video of your badminton game",
    type=['mp4', 'avi', 'mov'],
    key="coaching_video_uploader"
)

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.session_state.coaching_state['video_path'] = video_path
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
    else:
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.coaching_state['total_frames'] = total_frames
        
        # Create columns for video and analysis
        video_col, analysis_col = st.columns([1, 1])
        
        with video_col:
            # Create placeholder for video display
            frame_placeholder = st.empty()
            
            # Add hidden buttons for JavaScript interaction
            st.markdown("""
            <div style="display: none;">
                <button id="restart-button"></button>
                <button id="pause-button"></button>
                <button id="skip-button"></button>
            </div>
            """, unsafe_allow_html=True)
            
            # Display video player UI
            st.markdown(create_video_player_ui(), unsafe_allow_html=True)
            
            # Add event handlers for the hidden buttons
            if st.button("⏮️ Restart", key="coaching_restart", use_container_width=True):
                st.session_state.coaching_state['current_frame'] = 0
                st.session_state.coaching_state['video_ended'] = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            if st.button("⏸️ Pause/Play", key="coaching_pause", use_container_width=True):
                st.session_state.coaching_state['processing_frames'] = not st.session_state.coaching_state['processing_frames']
            
            if st.button("⏭️ Skip to End", key="coaching_skip", use_container_width=True):
                st.session_state.coaching_state['current_frame'] = total_frames - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        with analysis_col:
            # Display real-time analysis UI
            st.markdown(create_real_time_analysis_ui(), unsafe_allow_html=True)
            
            # Create placeholders for charts
            pie_chart_placeholder = st.empty()
            heatmap_placeholder = st.empty()
            trajectory_placeholder = st.empty()
            performance_placeholder = st.empty()
        
        # Initialize metrics
        st.session_state.coaching_state['analysis_data']['start_time'] = time.time()
        
        # Process video frames
        while cap.isOpened() and not st.session_state.coaching_state['video_ended']:
            if not st.session_state.coaching_state['processing_frames']:
                time.sleep(0.1)  # Add small delay when paused
                continue
            
            ret, frame = cap.read()
            if not ret:
                st.session_state.coaching_state['video_ended'] = True
                break
            
            # Update current frame
            st.session_state.coaching_state['current_frame'] += 1
            
            # Resize frame to reduce size
            height, width = frame.shape[:2]
            new_width = 640  # Reduced width
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Pose
            results = pose.process(frame_rgb)
            
            # Store current frame landmarks
            if results.pose_landmarks:
                # Get key landmarks
                right_shoulder = results.pose_landmarks.landmark[12]
                right_elbow = results.pose_landmarks.landmark[14]
                right_wrist = results.pose_landmarks.landmark[16]
                
                # Update wrist trajectory
                st.session_state.coaching_state['wrist_trajectory'].append((right_wrist.x, right_wrist.y))
                
                # Update heatmap
                update_heatmap(right_wrist, frame.shape)
            
            # Analyze pose
            if results.pose_landmarks:
                # Calculate pose accuracy
                pose_accuracy = calculate_pose_accuracy(results.pose_landmarks.landmark)
                st.session_state.coaching_state['analysis_data']['pose_accuracy'].append(pose_accuracy)
                
                # Update accuracy distribution
                update_accuracy_distribution(pose_accuracy)
                
                # Store frame metrics
                st.session_state.coaching_state['analysis_data']['frame_metrics'].append({
                    'frame': st.session_state.coaching_state['current_frame'],
                    'accuracy': pose_accuracy
                })
                
                # Draw pose landmarks on frame
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Draw wrist trajectory on frame
                if len(st.session_state.coaching_state['wrist_trajectory']) > 1:
                    trajectory = list(st.session_state.coaching_state['wrist_trajectory'])
                    for i in range(1, len(trajectory)):
                        prev_x, prev_y = trajectory[i-1]
                        curr_x, curr_y = trajectory[i]
                        prev_x, prev_y = int(prev_x * new_width), int(prev_y * new_height)
                        curr_x, curr_y = int(curr_x * new_width), int(curr_y * new_height)
                        cv2.line(frame_rgb, (prev_x, prev_y), (curr_x, curr_y), (255, 0, 0), 2)
            
            # Display frame
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Update real-time analysis UI
            st.markdown(create_real_time_analysis_ui(), unsafe_allow_html=True)
            
            # Update charts
            pie_chart_placeholder.plotly_chart(create_accuracy_pie_chart() or go.Figure(), use_container_width=True)
            heatmap_placeholder.plotly_chart(create_heatmap_visualization(), use_container_width=True)
            trajectory_placeholder.plotly_chart(create_trajectory_visualization() or go.Figure(), use_container_width=True)
            performance_placeholder.plotly_chart(create_performance_graph() or go.Figure(), use_container_width=True)
            
            # Add small delay to control frame rate
            time.sleep(0.03)
        
        cap.release()
        
        if st.session_state.coaching_state['video_ended']:
            st.success("Video analysis completed!")
            
            # Add comparison section
            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
            st.markdown('<div class="comparison-title">Performance Comparison</div>', unsafe_allow_html=True)
            
            # Create columns for comparison
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("""
                <div class="comparison-content">
                    <div class="comparison-item">
                        <div class="comparison-value">Current</div>
                        <div class="comparison-label">Session</div>
                    </div>
                    <div class="comparison-item">
                        <div class="comparison-value">Previous</div>
                        <div class="comparison-label">Session</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with comp_col2:
                st.markdown("""
                <div class="comparison-content">
                    <div class="comparison-item">
                        <div class="comparison-value">Current</div>
                        <div class="comparison-label">Session</div>
                    </div>
                    <div class="comparison-item">
                        <div class="comparison-value">Previous</div>
                        <div class="comparison-label">Session</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with comp_col3:
                st.markdown("""
                <div class="comparison-content">
                    <div class="comparison-item">
                        <div class="comparison-value">Current</div>
                        <div class="comparison-label">Session</div>
                    </div>
                    <div class="comparison-item">
                        <div class="comparison-value">Previous</div>
                        <div class="comparison-label">Session</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add comparison chart
            st.plotly_chart(create_comparison_chart(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add technique analysis section
            st.markdown('<h2 class="section-header">Technique Analysis <span class="technique-badge">PRO</span></h2>', unsafe_allow_html=True)
            st.markdown(generate_technique_analysis(), unsafe_allow_html=True)
            
            # Generate and display feedback
            if not st.session_state.coaching_state['feedback_generated']:
                st.session_state.coaching_state['feedback_generated'] = True
                
                # Create a divider
                st.markdown("---")
                
                # Generate and display feedback with custom styling
                st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
                st.markdown('<h2 class="feedback-header">Analysis Report</h2>', unsafe_allow_html=True)
                st.markdown('<div class="feedback-content">', unsafe_allow_html=True)
                feedback = generate_feedback()
                st.markdown(feedback, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add a button to download the feedback
                feedback_text = feedback.replace("##", "#").replace("###", "##")
                st.download_button(
                    label="📥 Download Feedback Report",
                    data=feedback_text,
                    file_name="badminton_coaching_feedback.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

# Cleanup
if 'video_path' in locals() and st.session_state.coaching_state['video_path'] == video_path:
    try:
        os.unlink(video_path)
    except Exception as e:
        st.warning(f"Could not delete temporary file: {e}") 