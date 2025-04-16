import cv2
import time
from rtmlib import Body, draw_skeleton # Make sure rtmlib is in the same directory or Python path
import config
from metrics import analyze_squat # Import analysis function

class VideoProcessor:
    def __init__(self, device='cpu', backend='onnxruntime', mode='balanced'):
        """
        Initializes the pose estimation model.

        Args:
            device (str): Device to run inference on ('cpu', 'cuda', 'mps').
            backend (str): Inference backend ('onnxruntime', 'opencv', 'openvino').
            mode (str): Model performance mode ('lightweight', 'balanced', 'performance').
        """
        print(f"Initializing RTMLib Body model with backend: {backend}, device: {device}, mode: {mode}")
        try:
            self.pose_model = Body(
                # Use pose='rtmo' explicitly if you want the one-stage model like in squat.py
                # pose='rtmo', # Uncomment if you prefer RTMO
                to_openpose=False, # Use COCO-17 format
                mode=mode,
                backend=backend,
                device=device
            )
            print("RTMLib Body model initialized successfully.")
        except Exception as e:
            print(f"Error initializing RTMLib model: {e}")
            raise

        self.keypoint_confidence_threshold = config.KEYPOINT_CONFIDENCE_THRESHOLD
        self.frame_data = [] # To store data per frame

    def process_video(self, video_path):
        """
        Processes the video file, performs pose estimation, and calculates metrics.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            tuple: (list of processed frames, list of metrics per frame)
                   Returns (None, None) if video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None, None, 0 # Return 0 fps on error
        
        # <<< Get video FPS >>>
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: # Handle case where FPS might not be readable
            print("Warning: Could not read video FPS. Defaulting GIF duration.")
            fps = 10 # Default to 10 FPS if unknown

        processed_frames_rgb = [] # Store frames in RGB format
        all_frame_metrics = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            start_time = time.time()

            # --- Pose Estimation ---
            try:
                # Assuming model returns keypoints for multiple people, take the first one
                keypoints, scores = self.pose_model(frame)
            except Exception as e:
                print(f"Error during pose model inference on frame {frame_idx}: {e}")
                # Optionally add a placeholder frame or skip
                # processed_frames_rgb.append(frame) # Add original frame on error
                all_frame_metrics.append({'frame': frame_idx, 'feedback': 'Inference Error'})
                frame_idx += 1
                continue # Skip analysis for this frame

            frame_metrics = {'frame': frame_idx}

            if keypoints.shape[0] > 0: # Check if any person was detected
                # Analyze the first detected person
                kpts = keypoints[0]
                scrs = scores[0]

                # --- Metric Calculation (Example: Squat) ---
                squat_metrics = analyze_squat(kpts, scrs)
                frame_metrics.update(squat_metrics) # Add squat metrics to frame data

                # --- Visualization ---
                # Draw skeleton on the frame
                img_show = draw_skeleton(frame.copy(), # Draw on a copy
                                         keypoints, # Pass all detected skeletons
                                         scores,
                                         openpose_skeleton=False, # Match model setting
                                         kpt_thr=self.keypoint_confidence_threshold,
                                         line_width=config.SKELETON_THICKNESS) # Use thickness from config

                # --- Add metric text with background ---
                exercise_text = f"Exercise: Squat"
                angle_text = f"Knee Angle: {frame_metrics.get('knee_angle', 'N/A')}"
                depth_text = f"Depth: {frame_metrics.get('squat_depth_feedback', 'N/A')}"
                texts_to_draw = [exercise_text, angle_text, depth_text] # Add more metrics here as needed

                # Define text thickness (make it slightly bolder for larger font)
                text_thickness = 3

                # Calculate position and draw background/text for each line
                text_y = config.TEXT_POSITION_OFFSET[1]
                for i, text in enumerate(texts_to_draw):
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, config.FONT, config.FONT_SCALE, text_thickness
                    )
                    # Calculate background rectangle coordinates
                    rect_x1 = config.TEXT_POSITION_OFFSET[0] - config.TEXT_PADDING
                    # Adjust y1 based on text_height correctly
                    rect_y1 = text_y - text_height - config.TEXT_PADDING
                    rect_x2 = config.TEXT_POSITION_OFFSET[0] + text_width + config.TEXT_PADDING
                    # Adjust y2 based on baseline
                    rect_y2 = text_y + baseline + config.TEXT_PADDING

                    # Draw black background rectangle
                    cv2.rectangle(img_show, (rect_x1, rect_y1), (rect_x2, rect_y2),
                                  config.TEXT_BG_COLOR, cv2.FILLED)

                    # Draw the text on top
                    cv2.putText(img_show, text, (config.TEXT_POSITION_OFFSET[0], text_y),
                                config.FONT, config.FONT_SCALE, config.FONT_COLOR,
                                text_thickness, cv2.LINE_AA) # Use text_thickness

                    # Update Y position for the next line
                    text_y += text_height + config.TEXT_LINE_SPACING + baseline # Add baseline for better spacing

            else:
                # No person detected
                img_show = frame.copy() # Show original frame
                frame_metrics['feedback'] = "No person detected"
                cv2.putText(img_show, "No person detected",
                           (config.TEXT_POSITION_OFFSET[0], config.TEXT_POSITION_OFFSET[1]),
                           config.FONT, config.FONT_SCALE, (0, 0, 255), 1, cv2.LINE_AA)

            # Processing time for frame
            processing_time = time.time() - start_time
            frame_metrics['processing_time'] = processing_time

            # print(f"Frame {frame_idx}: {processing_time:.4f}s, Metrics: {frame_metrics}") # Debug print

            # <<< Convert final frame to RGB and append >>>
            processed_frames_rgb.append(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
            all_frame_metrics.append(frame_metrics)
            frame_idx += 1

        cap.release()
        print(f"Video processing complete. Processed {frame_idx} frames. Original FPS: {fps:.2f}")
        # <<< Return frames in RGB and FPS >>>
        return processed_frames_rgb, all_frame_metrics, fps
