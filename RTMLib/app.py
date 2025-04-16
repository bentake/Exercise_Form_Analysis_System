import cv2
import streamlit as st
import tempfile
import os
import time
from pose_processor import VideoProcessor # Import processor class
import pandas as pd
import config
import imageio
import io # For bytes handling

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Exercise Form Analysis")
st.title("🏋️ Exercise Form Analysis Aid (MVP - Squat)")
st.caption("Upload a video of your squat for basic form analysis.")

# --- Sidebar for Options ---
st.sidebar.header("⚙️ Options")
# Add options later (e.g., model selection, thresholds)
selected_exercise = st.sidebar.selectbox("Select Exercise (MVP Focus: Squat)", ["Squat"]) #, "Deadlift", "Bench Press"]) # Add more later
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
        
        processing_done = False
        gif_bytes = None

        try:
            # Initialize processor (consider caching this for efficiency)
            # @st.cache_resource - Use caching if model loading is slow
            @st.cache_resource
            def get_video_processor(device, mode):
                 # Using 'onnxruntime' as default backend based on rtmlib examples
                return VideoProcessor(device=device, backend='onnxruntime', mode=mode)

            processor = get_video_processor(device_option, model_mode)

            start_process_time = time.time()
            # <<< Get RGB frames and FPS from processor >>>
            processed_frames_rgb, all_frame_metrics,fps = processor.process_video(video_path)
            total_process_time = time.time() - start_process_time

            if processed_frames_rgb is not None and len(processed_frames_rgb) > 0:
                st.success(f"Video processing finished in {total_process_time:.2f} seconds.")
                processing_done = True
                
                # # --- Display Processed Video ---
                # delay = 0.03 # Delay between frames for playback effect (adjust as needed)
                # stframe.info("Playing processed video...")
                # for frame in processed_frames_rgb:
                #     stframe.image(frame, channels="RGB", use_container_width=True)
                #     time.sleep(delay) # Simulate video playback speed
                # stframe.success("Playback finished.")

                # # --- Create GIF ---
                # with st.spinner("Creating GIF..."):
                #     gif_path = io.BytesIO() # Save GIF in memory
                #     # Calculate duration per frame for imageio (in seconds)
                #     duration = 1.0 / fps if fps > 0 else 0.1 # Default 100ms duration if fps is unknown
                #     # Limit FPS for smoother GIFs if original FPS is very high
                #     gif_fps_limit = 15
                #     if fps > gif_fps_limit:
                #         duration = 1.0 / gif_fps_limit

                #     imageio.mimsave(gif_path, processed_frames_rgb, format='GIF', duration=duration * 1000) # duration is in ms for imageio v3+
                #     gif_bytes = gif_path.getvalue()
                # stframe.success("GIF Created!")

                # --- Create GIF ---
                with st.spinner("Creating smaller GIF..."):
                    gif_path = io.BytesIO() # Save GIF in memory

                    # --- Add Resizing Logic ---
                    target_width = 720 # Adjust this target width as needed
                    resized_frames = []
                    if len(processed_frames_rgb) > 0:
                        # Get original dimensions from the first frame
                        h, w, _ = processed_frames_rgb[0].shape
                        aspect_ratio = h / w
                        target_height = int(target_width * aspect_ratio)
                        target_dim = (target_width, target_height)

                        for frame in processed_frames_rgb:
                            # Resize using OpenCV (expects BGR, but works on RGB too)
                            resized_frame = cv2.resize(frame, target_dim, interpolation=cv2.INTER_LINEAR)
                            resized_frames.append(resized_frame)
                    else:
                        resized_frames = processed_frames_rgb # Use original if no frames
                    # --- End Resizing Logic ---

                    # Calculate duration per frame for imageio (in seconds)
                    duration = 1.0 / fps if fps > 0 else 0.1 # Default 100ms duration if fps is unknown
                    gif_fps_limit = 30
                    if fps > gif_fps_limit:
                        duration = 1.0 / gif_fps_limit

                    # <<< Use resized_frames for mimsave >>>
                    imageio.mimsave(gif_path, resized_frames, format='GIF', duration=duration * 1000, loop=0) # duration is in ms for imageio v3+
                    gif_bytes = gif_path.getvalue()
                stframe.success("GIF Created!")


                # --- Display GIF ---
                stframe.image(gif_bytes, caption="Processed Analysis GIF", use_container_width=True)

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
                             summary_text += f"- Overall Feedback: Achieved good depth.\n"
                        else:
                             summary_text += f"- Overall Feedback: May need to go deeper.\n"
                    else:
                         summary_text += f"- Overall Feedback: Could not determine.\n"

                    summary_text += f"\n**Frame-by-Frame Data:**"
                    metrics_placeholder.markdown(summary_text)
                    # Rename columns before displaying the DataFrame
                    df_display = df_metrics[['frame', 'knee_angle', 'squat_depth_feedback']].rename(columns={
                        'frame': 'Frame',
                        'knee_angle': 'Knee Angle (°)',
                        'squat_depth_feedback': 'Squat Depth Feedback'
                    })
                    st.dataframe(df_display) # Display full data with new column names

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
        
        # --- Add Download Button if GIF was created ---
        if processing_done and gif_bytes:
            # Put download button in the main column below the GIF
            with col1:
                st.download_button(
                    label="Download Analysis GIF",
                    data=gif_bytes,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analysis.gif",
                    mime="image/gif"
                )

else:
    st.info("Upload a video file to begin analysis.")