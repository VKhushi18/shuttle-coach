import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Tuple

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define ideal angles for different shots
        self.ideal_angles = {
            "Smash": {
                "right_elbow": 165,  # Nearly straight arm
                "right_shoulder": 170,  # Raised high
                "right_hip": 160,  # Slight bend
                "right_knee": 150,  # Athletic stance
            },
            "Clear": {
                "right_elbow": 155,
                "right_shoulder": 160,
                "right_hip": 165,
                "right_knee": 155,
            },
            "Drop Shot": {
                "right_elbow": 100,  # More bent arm
                "right_shoulder": 130,
                "right_hip": 170,
                "right_knee": 160,
            },
            "Drive": {
                "right_elbow": 90,
                "right_shoulder": 90,
                "right_hip": 165,
                "right_knee": 155,
            },
            "Service": {
                "right_elbow": 120,
                "right_shoulder": 100,
                "right_hip": 175,
                "right_knee": 170,
            }
        }
        
    def detect_pose(self, image):
        """
        Detect pose landmarks in the image
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results
    
    def draw_pose(self, image, results):
        """
        Draw pose landmarks and connections on the image
        """
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def calculate_joint_angles(self, results) -> Dict[str, float]:
        """
        Calculate important joint angles for badminton technique analysis
        """
        if not results.pose_landmarks:
            return {}
            
        landmarks = results.pose_landmarks.landmark
        
        angles = {}
        
        # Right arm angles
        angles["right_shoulder"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        )
        
        angles["right_elbow"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
        )
        
        # Left arm angles
        angles["left_shoulder"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y)
        )
        
        angles["left_elbow"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y)
        )
        
        # Right leg angles
        angles["right_hip"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y)
        )
        
        angles["right_knee"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y),
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
             landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y)
        )
        
        # Left leg angles
        angles["left_hip"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y)
        )
        
        angles["left_knee"] = self._calculate_angle(
            (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y),
            (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
             landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y)
        )
        
        return angles
    
    def analyze_form(self, results, shot_type: str = None) -> List[str]:
        """
        Analyze player's form and return feedback based on shot type
        """
        feedback = []
        if not results.pose_landmarks:
            return ["No pose detected"]
        
        angles = self.calculate_joint_angles(results)
        
        # General posture checks
        spine_angle = self._calculate_spine_angle(results.pose_landmarks.landmark)
        if abs(spine_angle) > 20:
            feedback.append("Keep your back straight for better balance")
        
        # Shot-specific analysis
        if shot_type and shot_type in self.ideal_angles:
            ideal = self.ideal_angles[shot_type]
            
            for joint, ideal_angle in ideal.items():
                if joint in angles:
                    diff = abs(angles[joint] - ideal_angle)
                    if diff > 15:
                        feedback.append(f"Adjust your {joint.replace('_', ' ')} angle for better {shot_type}")
        
        # Check racket arm position
        if "right_elbow" in angles:
            if angles["right_elbow"] < 90:
                feedback.append("Extend your racket arm more for better reach")
        
        # Check ready position
        if "right_knee" in angles and "left_knee" in angles:
            if angles["right_knee"] > 160 and angles["left_knee"] > 160:
                feedback.append("Bend your knees more for better readiness")
        
        return feedback
    
    def calculate_posture_score(self, results) -> float:
        """
        Calculate a posture score based on key alignment factors
        """
        if not results.pose_landmarks:
            return 0.0
            
        score = 100.0
        landmarks = results.pose_landmarks.landmark
        
        # Check spine alignment
        spine_angle = self._calculate_spine_angle(landmarks)
        score -= abs(spine_angle) * 0.5  # Deduct points for spine tilt
        
        # Check shoulder alignment
        shoulder_tilt = abs(landmarks[11].y - landmarks[12].y) * 100
        score -= shoulder_tilt * 50  # Deduct points for uneven shoulders
        
        # Check hip alignment
        hip_tilt = abs(landmarks[23].y - landmarks[24].y) * 100
        score -= hip_tilt * 50  # Deduct points for uneven hips
        
        # Check knee bend
        right_knee_angle = self._calculate_angle(
            (landmarks[23].x, landmarks[23].y),
            (landmarks[25].x, landmarks[25].y),
            (landmarks[27].x, landmarks[27].y)
        )
        score -= abs(140 - right_knee_angle) * 0.2  # Ideal knee bend around 140 degrees
        
        return max(0, min(100, score))
    
    def _calculate_spine_angle(self, landmarks) -> float:
        """
        Calculate the angle of the spine relative to vertical
        """
        # Use nose and mid-hip points
        nose = landmarks[0]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate mid-hip point
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        
        return self._calculate_vertical_angle(
            (mid_hip_x, mid_hip_y),
            (nose.x, nose.y)
        )
    
    def _calculate_angle(self, point1: Tuple[float, float], 
                        point2: Tuple[float, float], 
                        point3: Tuple[float, float]) -> float:
        """
        Calculate angle between three points
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate dot product
        dot_product = np.dot(ba, bc)
        
        # Calculate magnitudes
        ba_mag = np.linalg.norm(ba)
        bc_mag = np.linalg.norm(bc)
        
        # Avoid division by zero
        if ba_mag == 0 or bc_mag == 0:
            return 0.0
            
        # Calculate cosine of angle
        cosine_angle = dot_product / (ba_mag * bc_mag)
        
        # Clamp to valid range for arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def _calculate_vertical_angle(self, point1: Tuple[float, float], 
                                point2: Tuple[float, float]) -> float:
        """
        Calculate angle from vertical
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return np.degrees(np.arctan2(dx, dy))
    
    def draw_advanced_pose(self, image, results, show_angles: bool = True) -> np.ndarray:
        """
        Draw advanced pose visualization with angles and guidelines
        """
        if not results.pose_landmarks:
            return image
            
        # Draw basic pose
        self.draw_pose(image, results)
        
        if show_angles:
            angles = self.calculate_joint_angles(results)
            landmarks = results.pose_landmarks.landmark
            
            # Draw angles on joints
            h, w = image.shape[:2]
            for joint, angle in angles.items():
                if "right_elbow" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
                elif "left_elbow" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
                elif "right_shoulder" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
                elif "left_shoulder" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
                elif "right_hip" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h))
                elif "left_hip" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
                elif "right_knee" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
                elif "left_knee" in joint:
                    pos = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y * h))
                else:
                    continue
                
                cv2.putText(image, f"{angle:.1f}°", pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image 