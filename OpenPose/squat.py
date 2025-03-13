import time
import cv2
from rtmlib import Body, draw_skeleton
import numpy as np

device = 'mps'  # Change to 'cuda' or 'mps' if available
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

cap = cv2.VideoCapture(0)
openpose_skeleton = True  # True for openpose-style, False for mmpose-style

body = Body(
    pose='rtmo',
    to_openpose=openpose_skeleton,
    mode='balanced',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0
squat_state = "standing"  # Can be "standing", "squatting", "rising"
squat_count = 0
previous_knee_angle = 180  # Initialize with a standing pose angle

# For stabilizing detection
knee_angles_buffer = []
buffer_size = 5

# Min knee angle to consider as squat
SQUAT_ANGLE_THRESHOLD = 110  # Adjust based on your squat depth preference

def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    if np.any(np.isnan([a, b, c])):
        return None
        
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure within valid range
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def is_valid_keypoint(keypoint, score, threshold=0.3):
    """Check if keypoint is valid based on score"""
    return score > threshold and not np.isnan(keypoint).any()

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1
    
    if not success:
        break
        
    s = time.time()
    keypoints, scores = body(frame)
    det_time = time.time() - s
    
    img_show = frame.copy()
    
    # Draw skeleton
    img_show = draw_skeleton(img_show,
                          keypoints,
                          scores,
                          openpose_skeleton=openpose_skeleton,
                          kpt_thr=0.3,
                          line_width=2)
    
    # Squat detection logic
    if keypoints.shape[0] > 0:  # If person detected
        # Get relevant keypoints (indices depend on model used)
        # For mmpose format (default):
        hip_idx, knee_idx, ankle_idx = 11, 12, 13  # Right side
        
        # For openpose format:
        if openpose_skeleton:
            hip_idx, knee_idx, ankle_idx = 8, 9, 10  # Right side
        
        hip = keypoints[0][hip_idx]
        knee = keypoints[0][knee_idx]
        ankle = keypoints[0][ankle_idx]
        
        hip_score = scores[0][hip_idx]
        knee_score = scores[0][knee_idx]
        ankle_score = scores[0][ankle_idx]
        
        # Only analyze if all keypoints are detected with good confidence
        if (is_valid_keypoint(hip, hip_score) and 
            is_valid_keypoint(knee, knee_score) and 
            is_valid_keypoint(ankle, ankle_score)):
            
            # Calculate knee angle
            knee_angle = calculate_angle(hip, knee, ankle)
            
            if knee_angle is not None:
                # Add to buffer for stabilization
                knee_angles_buffer.append(knee_angle)
                if len(knee_angles_buffer) > buffer_size:
                    knee_angles_buffer.pop(0)
                
                # Get smoothed angle
                avg_knee_angle = sum(knee_angles_buffer) / len(knee_angles_buffer)
                
                # Display angle
                cv2.putText(img_show, f"Knee angle: {avg_knee_angle:.1f}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Squat state machine
                if squat_state == "standing" and avg_knee_angle < SQUAT_ANGLE_THRESHOLD:
                    squat_state = "squatting"
                    
                elif squat_state == "squatting" and avg_knee_angle > 160:
                    squat_state = "standing"
                    squat_count += 1
                
                # Display squat count and state
                cv2.putText(img_show, f"Squats: {squat_count}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img_show, f"State: {squat_state}", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Resize and display
    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('Squat Detection', img_show)
    
    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()