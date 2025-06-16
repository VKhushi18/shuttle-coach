import numpy as np
import cv2
from collections import deque

class MovementAnalyzer:
    def __init__(self, max_positions=30):
        self.max_positions = max_positions
        self.player_positions = deque(maxlen=max_positions)
        self.court_zones = None
        self.movement_history = []
        
    def set_court_zones(self, frame_shape):
        """
        Define court zones for movement analysis
        """
        height, width = frame_shape[:2]
        
        # Define 6 zones (2x3 grid)
        self.court_zones = {
            'front_left': [(0, 0), (width//3, height//2)],
            'front_center': [(width//3, 0), (2*width//3, height//2)],
            'front_right': [(2*width//3, 0), (width, height//2)],
            'back_left': [(0, height//2), (width//3, height)],
            'back_center': [(width//3, height//2), (2*width//3, height)],
            'back_right': [(2*width//3, height//2), (width, height)]
        }
        
    def track_movement(self, player_position):
        """
        Track player movement and analyze patterns
        """
        self.player_positions.append(player_position)
        
        if len(self.player_positions) < 2:
            return None
            
        # Calculate movement metrics
        movement_vector = np.array(self.player_positions[-1]) - np.array(self.player_positions[-2])
        speed = np.linalg.norm(movement_vector)
        direction = np.arctan2(movement_vector[1], movement_vector[0])
        
        current_zone = self.get_current_zone(player_position)
        
        movement_data = {
            'speed': speed,
            'direction': direction,
            'zone': current_zone
        }
        
        self.movement_history.append(movement_data)
        return movement_data
    
    def get_current_zone(self, position):
        """
        Determine which court zone the player is in
        """
        if self.court_zones is None:
            return None
            
        x, y = position
        
        for zone_name, ((x1, y1), (x2, y2)) in self.court_zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
                
        return None
    
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
    
    def generate_heatmap(self, frame_shape):
        """
        Generate movement heatmap
        """
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        
        for pos in self.player_positions:
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(heatmap, (x, y), 20, 1.0, -1)
            
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    def _calculate_knee_angle(self, ankle, knee):
        """
        Calculate knee angle from ankle and knee positions
        """
        return np.degrees(np.arctan2(knee.y - ankle.y, knee.x - ankle.x)) 