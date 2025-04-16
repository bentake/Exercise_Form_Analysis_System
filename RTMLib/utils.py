import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees).

    Args:
        a (np.ndarray): Coordinates of the first point.
        b (np.ndarray): Coordinates of the second point (vertex).
        c (np.ndarray): Coordinates of the third point.

    Returns:
        float | None: Calculated angle in degrees, or None if input is invalid.
    """
    # Check for valid inputs (non-NaN)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return None

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate dot product and magnitudes
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return None

    # Calculate cosine of the angle
    cosine_angle = dot_product / (norm_ba * norm_bc)

    # Clip value to handle potential floating point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate angle in radians and convert to degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_valid_keypoint(keypoint, score, threshold=0.3):
    """Check if keypoint is valid based on score and NaN values."""
    return score > threshold and not np.any(np.isnan(keypoint))
