import numpy as np
import cv2

# Keypoint indices based on COCO-17 format (used by default in rtmlib Body)
# Check rtmlib/visualization/skeleton/coco17.py if needed
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# --- Squat Metrics Configuration ---
# Depth threshold: Hip below knee (relative Y coordinates)
# Example: hip.y > knee.y (assuming origin is top-left)
# We can also use angle: Hip-Knee-Ankle angle threshold
SQUAT_DEPTH_ANGLE_THRESHOLD = 95 # Angle in degrees (e.g., less than 95 means below parallel)

# Knee Valgus (requires front view ideally, rough side-view proxy below)
# Check if knee moves excessively inwards relative to hip-ankle line.
# This is a placeholder - proper valgus detection is complex from side view.
KNEE_VALGUS_THRESHOLD = 10 # Example threshold for angle deviation

# --- Thresholds ---
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3 # Minimum score to consider a keypoint valid

# --- Visualization ---
SKELETON_COLOR = (0, 255, 0) # Green skeleton
SKELETON_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.0 # <<< Increased font size significantly (adjust if needed)
FONT_COLOR = (64, 224, 208) # Turqoise color in BGR format
TEXT_POSITION_OFFSET = (15, 100) # <<< Adjusted offset for much larger font
TEXT_BG_COLOR = (0, 0, 0) # Black background
TEXT_PADDING = 8 # <<< Increased padding slightly
TEXT_LINE_SPACING = 15 # <<< Increased spacing between lines
