import numpy as np
import config
from utils import calculate_angle, is_valid_keypoint

def analyze_squat(keypoints, scores):
    """
    Analyzes squat form based on keypoints and scores.

    Args:
        keypoints (np.ndarray): Array of keypoints for a single person (num_keypoints, 2).
        scores (np.ndarray): Array of scores for the keypoints (num_keypoints,).

    Returns:
        dict: A dictionary containing calculated metrics and feedback.
              Returns empty dict if essential keypoints are missing.
    """
    metrics = {}

    # --- Get Keypoints ---
    # Use Right side for consistency (adjust if needed)
    right_hip = keypoints[config.RIGHT_HIP]
    right_knee = keypoints[config.RIGHT_KNEE]
    right_ankle = keypoints[config.RIGHT_ANKLE]
    # Left side for potential valgus check (though less reliable from side)
    # left_hip = keypoints[config.LEFT_HIP]
    # left_knee = keypoints[config.LEFT_KNEE]
    # left_ankle = keypoints[config.LEFT_ANKLE]

    # --- Get Scores ---
    hip_score = scores[config.RIGHT_HIP]
    knee_score = scores[config.RIGHT_KNEE]
    ankle_score = scores[config.RIGHT_ANKLE]

    # --- Check Keypoint Validity ---
    essential_points_valid = (
        is_valid_keypoint(right_hip, hip_score, config.KEYPOINT_CONFIDENCE_THRESHOLD) and
        is_valid_keypoint(right_knee, knee_score, config.KEYPOINT_CONFIDENCE_THRESHOLD) and
        is_valid_keypoint(right_ankle, ankle_score, config.KEYPOINT_CONFIDENCE_THRESHOLD)
    )

    if not essential_points_valid:
        metrics['feedback'] = "Missing essential keypoints (hip, knee, ankle)"
        return metrics # Cannot calculate primary metrics

    # --- Calculate Knee Angle ---
    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    metrics['knee_angle'] = knee_angle if knee_angle is not None else "N/A"

    # --- Calculate Squat Depth ---
    depth_feedback = "N/A"
    if knee_angle is not None:
        if knee_angle < config.SQUAT_DEPTH_ANGLE_THRESHOLD:
            depth_feedback = "Good Depth (Below Parallel)"
        else:
            depth_feedback = "Needs Improvement (Above Parallel)"
    metrics['squat_depth_feedback'] = depth_feedback

    # --- (Placeholder) Knee Valgus Check ---
    # This is a very basic check and unreliable from a side view.
    # A proper check requires a front view and measuring knee position
    # relative to the hip-ankle line or foot position.
    # For demonstration, we could check if knee X is too far 'inside' ankle X
    # knee_valgus_feedback = "N/A (Requires Front View)"
    # if is_valid_keypoint(left_knee, scores[config.LEFT_KNEE]): # Check if left side visible
    #     # Example basic check (likely inaccurate): Compare knee x to ankle x
    #     if right_knee[0] < right_ankle[0] - KNEE_VALGUS_THRESHOLD_PIXELS: # Example
    #          knee_valgus_feedback = "Potential Right Knee Valgus"
    # metrics['knee_valgus_feedback'] = knee_valgus_feedback
    metrics['knee_valgus_feedback'] = "N/A (Requires Front View)" # Defaulting to this

    metrics['feedback'] = "Analysis Complete" # General feedback if processing happened
    return metrics

# --- Add functions for Deadlift and Bench Press analysis later ---
# def analyze_deadlift(keypoints, scores):
#     metrics = {}
#     # ... implementation ...
#     return metrics

# def analyze_bench_press(keypoints, scores):
#     metrics = {}
#     # ... implementation ...
#     return metrics
