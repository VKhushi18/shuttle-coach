from ultralytics import YOLO
import cv2
import numpy as np
import torch

class ObjectTracker:
    def __init__(self, model_path=None):
        """
        Initialize object tracker with YOLOv8 model
        If model_path is None, use YOLOv8n as default
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
        
        # Classes we're interested in (from COCO dataset)
        self.target_classes = {
            32: 'sports ball',  # For shuttle
            39: 'bottle'        # Similar shape to racket, we'll fine-tune later
        }
        
    def detect_objects(self, frame):
        """
        Detect shuttle and racket in the frame
        """
        results = self.model(frame, stream=True)
        
        detections = {
            'shuttle': None,
            'racket': None
        }
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.target_classes:
                    conf = float(box.conf[0])
                    if conf > 0.3:  # Confidence threshold
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if self.target_classes[cls] == 'sports ball':
                            detections['shuttle'] = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': conf
                            }
                        else:
                            detections['racket'] = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': conf
                            }
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes for detected objects
        """
        for obj_type, detection in detections.items():
            if detection:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                
                color = (0, 255, 0) if obj_type == 'shuttle' else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{obj_type}: {conf:.2f}',
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, color, 2)
        
        return frame
    
    def analyze_trajectory(self, shuttle_positions, frame_shape):
        """
        Analyze shuttle trajectory from a list of positions
        """
        if len(shuttle_positions) < 2:
            return None
        
        # Calculate velocity and direction
        velocities = []
        for i in range(1, len(shuttle_positions)):
            prev_pos = shuttle_positions[i-1]
            curr_pos = shuttle_positions[i]
            
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities)
        
        # Determine shot type based on trajectory
        height_change = shuttle_positions[-1][1] - shuttle_positions[0][1]
        width_change = shuttle_positions[-1][0] - shuttle_positions[0][0]
        
        if height_change < 0 and avg_velocity > 50:
            return "Smash"
        elif height_change > 0 and abs(width_change) > frame_shape[1]/3:
            return "Clear"
        elif abs(height_change) < frame_shape[0]/4:
            return "Drive"
        
        return "Unknown shot" 