import numpy as np
from collections import deque
import math
import cv2

class MovementAnalyzer:
    def __init__(self):
        self.position_history = deque(maxlen=30)
        self.last_position = None
        self.court_zones = {
            'front': [(0, 0.33)],
            'mid': [(0.33, 0.66)],
            'back': [(0.66, 1.0)]
        }
        self.movement_history = []  # Store all positions for heatmap
    
    def track_movement(self, current_position):
        """
        Track player movement and calculate metrics
        """
        if self.last_position is None:
            self.last_position = current_position
            self.position_history.append(current_position)
            self.movement_history.append(current_position)  # Add to movement history
            return {'speed': 0, 'direction': 0, 'zone': self._get_court_zone(current_position)}
        
        # Calculate speed and direction
        dx = current_position[0] - self.last_position[0]
        dy = current_position[1] - self.last_position[1]
        
        speed = math.sqrt(dx*dx + dy*dy)
        direction = math.atan2(dy, dx)
        
        # Update history
        self.position_history.append(current_position)
        self.movement_history.append(current_position)  # Add to movement history
        self.last_position = current_position
        
        return {
            'speed': speed,
            'direction': direction,
            'zone': self._get_court_zone(current_position)
        }
    
    def _get_court_zone(self, position):
        """
        Determine which court zone the player is in
        """
        x, y = position
        relative_y = y / 480  # Normalize y position
        
        for zone, ranges in self.court_zones.items():
            for y_range in ranges:
                if y_range[0] <= relative_y < y_range[1]:
                    return zone
        return 'unknown'
    
    def is_service_motion(self, landmarks):
        """
        Detect if the player is performing a service motion based on pose landmarks
        """
        if not landmarks:
            return False
            
        # Get relevant joint positions
        right_shoulder = landmarks[12]  # MediaPipe pose landmark index for right shoulder
        right_elbow = landmarks[14]     # Right elbow
        right_wrist = landmarks[16]     # Right wrist
        right_hip = landmarks[24]       # Right hip
        
        # Calculate arm angle
        arm_angle = self._calculate_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y)
        )
        
        # Calculate shoulder to hip angle (verticality)
        vertical_angle = self._calculate_vertical_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_hip.x, right_hip.y)
        )
        
        # Service motion criteria:
        # 1. Arm should be relatively straight (angle > 150 degrees)
        # 2. Shoulder-hip line should be near vertical (angle < 20 degrees)
        is_arm_straight = arm_angle > 150
        is_body_vertical = abs(vertical_angle) < 20
        
        return is_arm_straight and is_body_vertical
    
    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        """
        # Convert points to vectors
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Calculate angle
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        return angle
    
    def _calculate_vertical_angle(self, p1, p2):
        """
        Calculate angle between a line and vertical
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate angle with vertical (90 degrees)
        angle = np.degrees(np.arctan2(dx, dy))
        
        return angle
    
    def get_movement_stats(self):
        """
        Calculate movement statistics from position history
        """
        if len(self.position_history) < 2:
            return None
            
        positions = np.array(self.position_history)
        
        # Calculate total distance
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        
        # Calculate average speed
        avg_speed = np.mean(distances)
        
        # Calculate court coverage
        x_coverage = np.ptp(positions[:, 0])
        y_coverage = np.ptp(positions[:, 1])
        
        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'court_coverage_x': x_coverage,
            'court_coverage_y': y_coverage
        }
    
    def generate_heatmap(self, frame_size):
        """
        Generate a heatmap of player movement patterns
        
        Args:
            frame_size (tuple): Size of the frame (height, width)
            
        Returns:
            numpy.ndarray: Heatmap visualization
        """
        if not self.movement_history:
            # Return empty heatmap if no movement data
            return np.zeros((*frame_size, 3), dtype=np.uint8)
        
        # Create empty heatmap
        heatmap = np.zeros((*frame_size, 3), dtype=np.uint8)
        
        # Convert positions to numpy array
        positions = np.array(self.movement_history)
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(
            positions[:, 0],  # x coordinates
            positions[:, 1],  # y coordinates
            bins=[32, 24],  # adjust bin size for smoothness
            range=[[0, frame_size[1]], [0, frame_size[0]]]
        )
        
        # Normalize histogram
        hist = hist / hist.max()
        
        # Scale histogram to image size
        hist_resized = cv2.resize(hist, (frame_size[1], frame_size[0]))
        
        # Create color mapping
        heatmap_colors = cv2.applyColorMap(
            (hist_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Add court lines
        court_lines = self._draw_court_lines(frame_size)
        
        # Blend heatmap with court lines
        alpha = 0.7
        heatmap = cv2.addWeighted(heatmap_colors, alpha, court_lines, 1-alpha, 0)
        
        return heatmap
    
    def _draw_court_lines(self, frame_size):
        """
        Draw badminton court lines for the heatmap
        """
        height, width = frame_size
        court = np.zeros((height, width, 3), dtype=np.uint8)
        
        return court
        
    def analyze_footwork(self, pose_landmarks):
        """
        Analyze player's footwork
        """
        if not pose_landmarks:
            return []
            
        feedback = []
        
        # Get ankle and knee positions
        left_ankle = pose_landmarks[27]
        right_ankle = pose_landmarks[28]
        left_knee = pose_landmarks[25]
        right_knee = pose_landmarks[26]
        
        # Check stance width
        ankle_distance = np.linalg.norm(
            np.array([left_ankle.x, left_ankle.y]) - 
            np.array([right_ankle.x, right_ankle.y])
        )
        
        if ankle_distance < 0.1:  # Too narrow
            feedback.append("Widen your stance for better stability")
        elif ankle_distance > 0.5:  # Too wide
            feedback.append("Narrow your stance for better mobility")
            
        # Check knee bend
        left_knee_angle = self._calculate_knee_angle(left_ankle, left_knee)
        right_knee_angle = self._calculate_knee_angle(right_ankle, right_knee)
        
        if min(left_knee_angle, right_knee_angle) > 160:
            feedback.append("Bend your knees more for better readiness")
            
        return feedback

    def calculate_footwork_score(self, landmarks):
        """
        Calculate a score based on the player's footwork
        """
        try:
            if not landmarks or len(landmarks) < 33:
                return 0.0
            
            # Get ankle and knee positions
            right_ankle = landmarks[28]
            left_ankle = landmarks[27]
            right_knee = landmarks[26]
            left_knee = landmarks[25]
            
            # Calculate stance width (distance between ankles)
            stance_width = self._calculate_distance(
                (right_ankle.x, right_ankle.y),
                (left_ankle.x, left_ankle.y)
            )
            
            # More lenient stance width check
            if stance_width < 0.2:  # Reduced minimum stance width
                stance_score = 50  # Increased base score
            elif stance_width > 0.8:  # Increased maximum stance width
                stance_score = 50  # Increased base score
            else:
                stance_score = 100
            
            # Calculate knee angles
            right_knee_angle = self._calculate_knee_angle(right_ankle, right_knee)
            left_knee_angle = self._calculate_knee_angle(left_ankle, left_knee)
            
            # More lenient knee bend check
            if right_knee_angle > 160 and left_knee_angle > 160:  # Increased angle threshold
                knee_score = 50  # Increased base score
            else:
                knee_score = 100
            
            # Weight the scores (stance is more important)
            return (stance_score * 0.6) + (knee_score * 0.4)
            
        except Exception as e:
            print(f"Error in calculate_footwork_score: {e}")
            return 0.0

    def detect_shot_type(self, pose_landmarks):
        """
        Detect the type of shot being played based on pose landmarks
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            str: Type of shot detected ('smash', 'clear', 'drop', 'drive', or None)
        """
        try:
            if not pose_landmarks or len(pose_landmarks) < 33:
                return None
                
            # Get key landmarks
            right_shoulder = pose_landmarks[12]
            right_elbow = pose_landmarks[14]
            right_wrist = pose_landmarks[16]
            right_hip = pose_landmarks[24]
            
            # Check if landmarks are visible
            if not all([right_shoulder, right_elbow, right_wrist, right_hip]):
                return None
                
            # Calculate wrist position relative to shoulder
            wrist_height = right_wrist.y - right_shoulder.y
            
            # Calculate movement speed
            if len(self.position_history) >= 3:
                recent_positions = list(self.position_history)[-3:]
                speeds = []
                for i in range(1, len(recent_positions)):
                    speed = self._calculate_distance(recent_positions[i-1], recent_positions[i])
                    speeds.append(speed)
                avg_speed = sum(speeds) / len(speeds)
            else:
                avg_speed = 0
            
            # More lenient shot detection based on wrist position and movement
            if wrist_height < 0:  # Wrist above or at shoulder level (more lenient)
                if avg_speed > 2:  # Reduced speed threshold
                    return 'smash'
                else:
                    return 'clear'
            else:  # Wrist below shoulder
                if avg_speed < 3:  # Reduced speed threshold
                    return 'drop'
                else:
                    return 'drive'
            
            return None
        except Exception as e:
            print(f"Error in detect_shot_type: {e}")
            return None

    def _calculate_knee_angle(self, ankle, knee):
        """
        Calculate knee angle based on ankle and knee positions
        """
        dx = knee.x - ankle.x
        dy = knee.y - ankle.y
        return np.degrees(np.arctan2(dy, dx))

    def _calculate_distance(self, p1, p2):
        """
        Calculate Euclidean distance between two points
        """
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) 