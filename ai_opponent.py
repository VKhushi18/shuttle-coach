import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Dict
import random
import math

@dataclass
class ShotPattern:
    name: str
    target_zones: List[Tuple[float, float]]  # x, y coordinates as percentage of court
    speed: float  # shot speed
    trajectory: float  # trajectory angle in degrees

class AIOpponent:
    def __init__(self, difficulty='intermediate'):
        self.difficulty = difficulty
        self.court_width = 640
        self.court_height = 480
        self.position = (self.court_width // 2, self.court_height // 4)  # AI starts at top of court
        self.target_position = self.position
        self.speed = self._get_difficulty_params()['speed']
        self.reaction_time = self._get_difficulty_params()['reaction_time']
        self.accuracy = self._get_difficulty_params()['accuracy']
        
        # Shuttle properties
        self.shuttle_pos = None
        self.shuttle_velocity = (0, 0)
        self.shuttle_in_play = False
        self.gravity = 9.8
        self.air_resistance = 0.1
        
        # Define common shot patterns
        self.shot_patterns = {
            'clear': ShotPattern(
                name='clear',
                target_zones=[(0.2, 0.9), (0.8, 0.9)],  # back corners
                speed=0.8,
                trajectory=60
            ),
            'drop': ShotPattern(
                name='drop',
                target_zones=[(0.3, 0.3), (0.7, 0.3)],  # front corners
                speed=0.5,
                trajectory=45
            ),
            'smash': ShotPattern(
                name='smash',
                target_zones=[(0.4, 0.6), (0.6, 0.6)],  # mid-court
                speed=1.0,
                trajectory=30
            ),
            'drive': ShotPattern(
                name='drive',
                target_zones=[(0.3, 0.5), (0.7, 0.5)],  # mid-court sides
                speed=0.7,
                trajectory=15
            )
        }
        
        # Training patterns for different aspects
        self.training_patterns = {
            'footwork': [
                ('clear', 'drop', 'clear', 'drop'),  # front-back movement
                ('drive', 'drive', 'drive', 'drive')  # side-to-side movement
            ],
            'reaction': [
                ('smash', 'drop', 'smash', 'clear'),  # varied speed shots
                ('drive', 'smash', 'drop', 'drive')   # mixed patterns
            ],
            'endurance': [
                ('clear', 'clear', 'clear', 'smash'),  # long rallies
                ('drive', 'drive', 'smash', 'drop')    # continuous movement
            ]
        }
    
    def _get_difficulty_params(self) -> Dict:
        """Return parameters based on difficulty level"""
        params = {
            'beginner': {
                'speed': 0.5,
                'reaction_time': 1.0,
                'accuracy': 0.6,
                'shot_speed_multiplier': 0.7
            },
            'intermediate': {
                'speed': 0.7,
                'reaction_time': 0.7,
                'accuracy': 0.8,
                'shot_speed_multiplier': 0.85
            },
            'advanced': {
                'speed': 0.9,
                'reaction_time': 0.4,
                'accuracy': 0.9,
                'shot_speed_multiplier': 1.0
            },
            'expert': {
                'speed': 1.0,
                'reaction_time': 0.3,
                'accuracy': 0.95,
                'shot_speed_multiplier': 1.2
            }
        }
        return params.get(self.difficulty, params['intermediate'])
    
    def hit_shuttle(self, target_pos: Tuple[int, int], shot_type: str):
        """Simulate hitting the shuttle towards a target position"""
        if not self.shuttle_in_play:
            self.shuttle_pos = self.position
            self.shuttle_in_play = True
        
        # Calculate direction and speed based on shot type
        pattern = self.shot_patterns[shot_type]
        dx = target_pos[0] - self.position[0]
        dy = target_pos[1] - self.position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Base speed depends on shot type and difficulty
        base_speed = pattern.speed * self._get_difficulty_params()['shot_speed_multiplier'] * 20
        
        # Calculate initial velocity components
        angle = math.atan2(dy, dx)
        trajectory_angle = math.radians(pattern.trajectory)
        
        self.shuttle_velocity = (
            base_speed * math.cos(angle),
            base_speed * math.sin(angle) - base_speed * math.sin(trajectory_angle)
        )
    
    def update_shuttle(self, dt: float):
        """Update shuttle position based on physics"""
        if self.shuttle_in_play and self.shuttle_pos:
            # Update velocity with gravity and air resistance
            vx, vy = self.shuttle_velocity
            vy += self.gravity * dt
            
            # Apply air resistance
            speed = math.sqrt(vx*vx + vy*vy)
            resistance = self.air_resistance * speed * speed
            if speed > 0:
                vx -= (vx/speed) * resistance * dt
                vy -= (vy/speed) * resistance * dt
            
            self.shuttle_velocity = (vx, vy)
            
            # Update position
            x, y = self.shuttle_pos
            x += vx * dt
            y += vy * dt
            
            # Check court boundaries
            if x < 0 or x > self.court_width:
                self.shuttle_in_play = False
            if y < 0 or y > self.court_height:
                self.shuttle_in_play = False
            
            self.shuttle_pos = (int(x), int(y))
    
    def get_next_shot(self, player_position: Tuple[int, int], shuttle_position: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
        """Determine the next shot based on player position and shuttle position"""
        # Update shuttle position
        self.shuttle_pos = shuttle_position
        
        # Calculate distances
        player_distance = np.sqrt(
            (player_position[0] - shuttle_position[0])**2 +
            (player_position[1] - shuttle_position[1])**2
        )
        
        # Choose appropriate shot based on position and distance
        if player_distance > self.court_height * 0.7:  # Player far from shuttle
            shot_type = 'drop'  # Force player to run forward
        elif player_position[1] < self.court_height * 0.3:  # Player in front
            shot_type = 'clear'  # Push player back
        elif random.random() < 0.3:  # 30% chance for aggressive shot
            shot_type = 'smash'
        else:
            shot_type = random.choice(['drive', 'clear', 'drop'])
        
        # Get target position with some randomness based on accuracy
        pattern = self.shot_patterns[shot_type]
        target_zone = random.choice(pattern.target_zones)
        target_x = int(target_zone[0] * self.court_width)
        target_y = int(target_zone[1] * self.court_height)
        
        # Add randomness based on accuracy
        accuracy_variance = 1 - self.accuracy
        target_x += int(random.uniform(-50, 50) * accuracy_variance)
        target_y += int(random.uniform(-50, 50) * accuracy_variance)
        
        # Ensure target stays within court bounds
        target_x = np.clip(target_x, 0, self.court_width)
        target_y = np.clip(target_y, 0, self.court_height)
        
        # Hit the shuttle
        self.hit_shuttle((target_x, target_y), shot_type)
        
        return shot_type, (target_x, target_y)
    
    def get_training_sequence(self, focus_area: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Generate a training sequence focused on a specific area of improvement
        """
        pattern = random.choice(self.training_patterns.get(focus_area, []))
        sequence = []
        
        for shot_type in pattern:
            shot_pattern = self.shot_patterns[shot_type]
            target_zone = random.choice(shot_pattern.target_zones)
            target_x = int(target_zone[0] * self.court_width)
            target_y = int(target_zone[1] * self.court_height)
            sequence.append((shot_type, (target_x, target_y)))
        
        return sequence
    
    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw AI opponent and shuttle on the frame"""
        # Draw AI player position
        cv2.circle(frame, self.position, 20, (0, 0, 255), -1)  # Red circle for AI
        cv2.putText(frame, f"AI ({self.difficulty})", 
                   (self.position[0] - 30, self.position[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw shuttle if in play
        if self.shuttle_in_play and self.shuttle_pos:
            cv2.circle(frame, self.shuttle_pos, 5, (255, 255, 255), -1)  # White circle for shuttle
            # Draw shuttle trail
            vx, vy = self.shuttle_velocity
            end_pos = (
                int(self.shuttle_pos[0] + vx * 0.1),
                int(self.shuttle_pos[1] + vy * 0.1)
            )
            cv2.line(frame, self.shuttle_pos, end_pos, (255, 255, 255), 1)
        
        return frame
    
    def update_position(self, target_position: Tuple[int, int], dt: float):
        """Update AI position based on target position and time delta"""
        dx = target_position[0] - self.position[0]
        dy = target_position[1] - self.position[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 1:
            move_x = dx / distance * self.speed * dt * 200  # Increased movement speed
            move_y = dy / distance * self.speed * dt * 200
            self.position = (
                int(self.position[0] + move_x),
                int(self.position[1] + move_y)
            )
            
        # Update shuttle physics
        self.update_shuttle(dt) 