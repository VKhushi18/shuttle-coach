import cv2
import numpy as np
import plotly.graph_objects as go
from collections import deque

class Visualizer:
    def __init__(self):
        self.court_lines = self._initialize_court_lines()
        self.motion_trail = []
        self.trail_length = 30
        
    def _initialize_court_lines(self):
        # Standard badminton court dimensions (in pixels for 640x480 frame)
        return {
            'outer': [(100, 50), (540, 50), (540, 430), (100, 430)],
            'service_line': [(100, 140), (540, 140)],
            'center_line': [(320, 50), (320, 430)],
            'doubles_side': [(50, 50), (590, 50), (590, 430), (50, 430)]
        }
    
    def draw_court(self, frame):
        # Draw court lines
        for line_points in self.court_lines.values():
            for i in range(len(line_points)-1):
                cv2.line(frame, line_points[i], line_points[i+1], (255, 255, 255), 2)
        return frame
    
    def draw_motion_trails(self, frame, shuttle_pos, player_pos):
        if shuttle_pos:
            self.motion_trail.append(shuttle_pos)
            if len(self.motion_trail) > self.trail_length:
                self.motion_trail.pop(0)
            
            # Draw motion trail with fading effect
            for i in range(len(self.motion_trail)-1):
                alpha = i / len(self.motion_trail)
                color = (0, int(255 * alpha), int(255 * (1-alpha)))
                cv2.line(frame, self.motion_trail[i], self.motion_trail[i+1], color, 2)
        return frame
    
    def overlay_feedback(self, frame, feedback_list):
        y_offset = 50
        for feedback in feedback_list:
            cv2.putText(frame, feedback, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
        return frame
    
    def create_movement_plot(self, movement_history):
        if not movement_history:
            return None
            
        x_coords = [pos[0] for pos in movement_history]
        y_coords = [pos[1] for pos in movement_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords,
                               mode='lines+markers',
                               name='Movement Path'))
        
        fig.update_layout(
            title="Player Movement Analysis",
            xaxis_title="Court Width",
            yaxis_title="Court Length",
            showlegend=True
        )
        return fig
    
    def draw_game_overlay(self, frame, player_score, opponent_score, current_server, service_court):
        # Draw score
        score_text = f"{player_score} - {opponent_score}"
        cv2.putText(frame, score_text, (280, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw server indicator
        server_text = f"Server: {'Player' if current_server == 'player' else 'Opponent'}"
        cv2.putText(frame, server_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Highlight current service court
        service_box = self._get_service_box(service_court)
        cv2.rectangle(frame, service_box[0], service_box[1], 
                     (0, 255, 0), 2)
        
        return frame
    
    def _get_service_box(self, service_court):
        if service_court == 'right':
            return [(320, 140), (540, 50)]  # Right service court
        else:
            return [(100, 140), (320, 50)]  # Left service court
    
    def is_shuttle_out(self, shuttle_pos):
        x, y = shuttle_pos
        # Check if shuttle is outside court boundaries
        return (x < 50 or x > 590 or  # Side boundaries
                y < 50 or y > 430)    # End boundaries
    
    def validate_service(self, player_pos, service_court):
        x, y = player_pos
        if service_court == 'right':
            return 320 <= x <= 540 and 140 <= y <= 430
        else:
            return 100 <= x <= 320 and 140 <= y <= 430
    
    def draw_advanced_pose(self, frame, pose_landmarks, angles):
        if not pose_landmarks:
            return frame
            
        # Draw joint angles
        for joint, angle in angles.items():
            joint_point = pose_landmarks[joint]
            x = int(joint_point.x * frame.shape[1])
            y = int(joint_point.y * frame.shape[0])
            cv2.putText(frame, f"{angle:.1f}°", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame 