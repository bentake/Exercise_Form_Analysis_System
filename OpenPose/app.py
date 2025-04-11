import streamlit as st
import cv2
import tempfile
import os
import time
from pose_processor import VideoProcessor # Import your processor class
import pandas as pd
import config

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Weightlifting Form Analysis")
st.title("🏋️ Weightlifting Form Analysis Aid (MVP - Squat)")
st.caption("Upload a video of your squat for basic form analysis.")

# --- Sidebar for Options ---
st.sidebar.header("⚙️ Options")
# Add options later (e.g., model selection, thresholds)
selected_lift = st.sidebar.selectbox("Select Lift (MVP Focus: Squat)", ["Squat"]) #, "Deadlift", "Bench Press"]) # Add more later
device_option = st.sidebar.selectbox("Select Compute Device", ["cpu", "cuda", "mps"], help="Select 'cuda' or 'mps' if you have compatible hardware and drivers installed.")
model_mode = st.sidebar.selectbox("Select Model Mode", ["balanced", "lightweight", "performance"], index=0, help="Balanced: Good speed/accuracy. Lightweight: Faster, less accurate. Performance: Slower, more accurate.")

# --- Main Area ---
uploaded_file = st.file_uploader("Choose a video file (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.success(f"Video '{uploaded_file.name}' uploaded successfully.")

        # --- Processing ---
        col1, col2 = st.columns([2, 1]) # Column for video, column for metrics

        with col1:
            stframe = st.empty() # Placeholder for video frames
            stframe.info("Processing video... Please wait.")

        with col2:
            st.subheader("📊 Analysis Results")
            metrics_placeholder = st.empty() # Placeholder for metrics summary


        try:
            # Initialize processor (consider caching this for efficiency)
            # @st.cache_resource - Use caching if model loading is slow
            @st.cache_resource
            def get_video_processor(device, mode):
                 # Using 'onnxruntime' as default backend based on rtmlib examples
                return VideoProcessor(device=device, backend='onnxruntime', mode=mode)

            processor = get_video_processor(device_option, model_mode)

            start_process_time = time.time()
            processed_frames, all_frame_metrics = processor.process_video(video_path)
            total_process_time = time.time() - start_process_time

            if processed_frames is not None:
                st.success(f"Video processing finished in {total_process_time:.2f} seconds.")

                # --- Display Processed Video ---
                delay = 0.03 # Delay between frames for playback effect (adjust as needed)
                stframe.info("Playing processed video...")
                for frame in processed_frames:
                    stframe.image(frame, channels="BGR", use_container_width=True)
                    time.sleep(delay) # Simulate video playback speed
                stframe.success("Playback finished.")


                # --- Display Metrics Summary ---
                if all_frame_metrics:
                    df_metrics = pd.DataFrame(all_frame_metrics)

                    # Example Summary: Find min knee angle, check depth overall
                    min_knee_angle = df_metrics['knee_angle'].replace("N/A", float('inf')).min()
                    avg_proc_time = df_metrics['processing_time'].mean()

                    summary_text = f"**Summary:**\n"
                    summary_text += f"- Minimum Knee Angle: {min_knee_angle:.1f}°\n" if min_knee_angle != float('inf') else "- Minimum Knee Angle: N/A\n"
                    summary_text += f"- Avg. Frame Processing Time: {avg_proc_time:.3f}s\n"

                    # Check overall depth based on min angle
                    if min_knee_angle != float('inf'):
                        # Use config.SQUAT_DEPTH_ANGLE_THRESHOLD here
                        if min_knee_angle < config.SQUAT_DEPTH_ANGLE_THRESHOLD:
                             summary_text += f"- Overall Depth: Achieved good depth.\n"
                        else:
                             summary_text += f"- Overall Depth: May need to go deeper.\n"
                    else:
                         summary_text += f"- Overall Depth: Could not determine.\n"

                    summary_text += f"\n**Frame-by-Frame Data:**"
                    metrics_placeholder.markdown(summary_text)
                    st.dataframe(df_metrics[['frame', 'knee_angle', 'squat_depth_feedback', 'feedback', 'processing_time']]) # Display full data

                else:
                    metrics_placeholder.warning("No metrics were generated during processing.")

            else:
                st.error("Failed to process video.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            # Print traceback for debugging if needed
            import traceback
            st.error(traceback.format_exc())
        finally:
            # Clean up temporary file
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
                # print(f"Removed temporary file: {video_path}") # Debug print

else:
    st.info("Upload a video file to begin analysis.")