# Exercise Form Analysis System

This project is a Streamlit web application designed to provide basic analysis of exercise form using computer vision. It leverages the `rtmlib` library for pose estimation based on RTMPose/RTMO models.

This is currently a Minimum Viable Product (MVP) focused primarily on analyzing Squat videos.

## Current Features (MVP v1.0)

* **Video Upload:** Accepts common video formats (.mp4, .mov, .avi).
* **Squat Analysis:** Analyzes uploaded videos specifically identified as squats.
* **Pose Estimation:** Uses `rtmlib` (RTMPose/RTMO models via ONNX Runtime) to detect body keypoints frame-by-frame.
* **Basic Metrics:**
    * Calculates knee angle throughout the movement.
    * Provides basic squat depth feedback based on a configurable knee angle threshold (`SQUAT_DEPTH_ANGLE_THRESHOLD` in `config.py`).
* **Visualization:**
    * Overlays detected skeletons on video frames.
    * Displays calculated knee angle and depth feedback directly on the video frames.
    * Allows customization of text color, size, and background for visibility (see `config.py`).
* **GIF Output:** Generates, displays, and allows downloading of an animated GIF showing the processed video with overlays. Options are provided in the code (`app.py`) to adjust GIF resolution and frame rate to manage file size.
* **Results Display:**
    * Shows the generated GIF analysis.
    * Provides a basic summary (e.g., Minimum Knee Angle achieved).
    * Displays a table with frame-by-frame metrics (knee angle, depth achieved flag, processing time).
* **Configuration:**
    * Select compute device (CPU, CUDA, MPS) via the UI sidebar.
    * Select model performance mode (balanced, lightweight, performance) via the UI sidebar.

## Project Structure

```
RTMLib/
├── rtmlib/             # Included rtmlib library for pose estimation
│   ├── ...
├── venv/               # Python virtual environment (created by user)
├── app.py              # Main Streamlit application script
├── config.py           # Configuration for keypoint indices, thresholds, visualization
├── metrics.py          # Functions for calculating exercise metrics (currently squat)
├── pose_processor.py   # Handles video processing, pose estimation, and visualization
├── requirements.txt    # Project dependencies
├── utils.py            # Utility functions (e.g., angle calculation)
└── README.md           # This file
```


## Setup Instructions

1.  **Prerequisites:**
    * Python 3.8+
    * Git (optional, for cloning)

2.  **Get the Code:**
    * Clone the repository or download the source code.
    * Ensure the `rtmlib` folder is present within the main project directory (`RTMLib/`).

3.  **Create Virtual Environment:**
    * Open your terminal or command prompt, navigate to the project directory (`RTMLib/`).
    * Create a virtual environment:
        ```bash
        python -m venv venv
        ```
    * Activate the environment:
        * Windows: `venv\Scripts\activate`
        * macOS/Linux: `source venv/bin/activate`

4.  **Install Dependencies:**
    * Make sure your `requirements.txt` file includes:
        ```
        numpy
        onnxruntime # Or onnxruntime-gpu if using CUDA
        opencv-contrib-python
        opencv-python
        tqdm
        streamlit
        pandas
        imageio
        ```
    * Install the requirements:
        ```bash
        pip install -r requirements.txt
        ```
    * **Note on ONNX Runtime:** If you have a compatible NVIDIA GPU and CUDA installed, install `onnxruntime-gpu` instead of `onnxruntime` for potentially faster processing when selecting the 'cuda' device option. For Apple Silicon Macs, `onnxruntime` includes CoreML support which can be used with the 'mps' device option.

## How to Run

1.  Activate your virtual environment (if not already active).
2.  Navigate to the project directory (`RTMLib/`) in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application should open in your web browser.

## Usage

1.  **Select Options:** Use the sidebar to choose the compute device and model performance mode. (Note: Lift selection is currently limited to Squat).
2.  **Upload Video:** Click "Browse files" and select a video file of a squat performance.
3.  **Processing:** Wait for the application to process the video. Status messages will be displayed.
4.  **View Results:**
    * An animated GIF of the processed video with skeleton and metric overlays will be displayed in the left column.
    * A summary analysis (currently basic, showing minimum knee angle) will appear in the right column.
    * A detailed table with frame-by-frame metrics can be viewed by expanding the section below the summary.
    * A download button for the GIF will appear below the GIF display.

## Configuration

You can adjust analysis and visualization parameters by editing the `config.py` file:

* `SQUAT_DEPTH_ANGLE_THRESHOLD`: Modify the knee angle threshold for determining squat depth.
* Visualization constants (`FONT_SCALE`, `FONT_COLOR`, `TEXT_BG_COLOR`, etc.): Change the appearance of the overlay text.
* Keypoint indices: These are currently set for the COCO-17 format used by `rtmlib`.

## Limitations & Future Work

* **MVP Stage:** This is an early version focused on demonstrating the core pipeline for squats.
* **Squat Only:** Analysis logic for Deadlift and Bench Press is not yet implemented.
* **Basic Metrics:** Only knee angle and a simple depth check are calculated. More metrics (back angle, bar path, torso lean, stability) are needed for comprehensive analysis.
* **Simple Summary:** The current summary analysis is very basic. More sophisticated aggregation and interpretation (including potential LLM integration) are planned.
* **Rep Counting:** Basic rep counting is not fully integrated or robust.
* **Viewpoint Dependency:** Analysis assumes a relatively clear side view for accurate angle calculation. Frontal or angled views may produce less reliable results for current metrics.
* **Error Handling:** Basic error handling exists, but can be improved.

**Planned Features:**

* Implement analysis modules (`metrics.py`) for Deadlift and Bench Press.
* Add more relevant metrics (back angle, bar path, torso angle, joint velocities).
* Implement robust rep counting.
* Develop more sophisticated summary logic and potentially integrate LLM-based feedback.
* Improve handling of different camera angles and potential occlusions.
* User profiles and progress tracking.

## Acknowledgements

* This project relies heavily on the lightweight [rtmlib](https://github.com/Tau-J/rtmlib) library by Tau-J.
* Pose estimation is performed using models based on RTMPose/RTMO developed by members of OpenMMLab.

    ```
    @misc{jiang2023,
      doi = {10.48550/ARXIV.2303.07399},
      url = {[https://arxiv.org/abs/2303.07399](https://arxiv.org/abs/2303.07399)},
      author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
      title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
      publisher = {arXiv},
      year = {2023},
    }

    @misc{lu2023rtmo,
          title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
          author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
          year={2023},
          eprint={2312.07526},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    ```
