# CCA 3 Computer Vision Assignment

**Candidate:** Himanshu Jaiswal | **PRN:** 1272250599 | **Course:** Computer Vision  

A comprehensive, reproducible, and Colab-compatible academic project implementing classical and modern Computer Vision algorithms across Image Restoration, Segmentation, Feature Extraction, Object Recognition, and Video Motion Estimation.

Developed for Master of Computer Applications (MCA) / Graduate Computer Vision coursework.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Assignment Objectives](#assignment-objectives)
3. [Summary of the Four Tasks](#summary-of-the-four-tasks)
   - [Task 1: Image Restoration and Segmentation](#task-1-image-restoration-and-segmentation)
   - [Task 2: Feature Extraction and Matching](#task-2-feature-extraction-and-matching)
   - [Task 3: Object Recognition](#task-3-object-recognition)
   - [Task 4: Video Motion Estimation](#task-4-video-motion-estimation)
4. [Technologies and Libraries Used](#technologies-and-libraries-used)
5. [Project Directory Structure](#project-directory-structure)
6. [Supplying Your Own Input Data](#supplying-your-own-input-data)
   - [Local Method: Data Directory](#local-method-data-directory)
   - [Google Colab Direct Upload](#google-colab-direct-upload)
7. [Installation & Setup Instructions](#installation--setup-instructions)
   - [Local Windows Execution](#local-windows-execution)
   - [Google Colab Execution](#google-colab-execution)
8. [Explanation of Output Directories](#explanation-of-output-directories)
9. [Running Tests and Algorithmic Validation](#running-tests-and-algorithmic-validation)
10. [Project Limitations](#project-limitations)
11. [Academic Integrity Statement](#academic-integrity-statement)

---

## Project Overview
This repository contains the complete implementation for the **CCA 3 Computer Vision Assignment**. All algorithm implementations, mathematical explanations, visualizations, and comparative metric calculations are contained directly within a single Google Colab-compatible Jupyter notebook:

`CCA_3_Computer_Vision_Assignment.ipynb`

No external source scripts or `src/` modules are required. The notebook executes from top to bottom, automatically creates required directories, checks input data, and exports quantitative metrics and visualization figures to organized output folders.

---

## Assignment Objectives
- Formulate and analyze mathematical degradation models for visual data (Additive White Gaussian Noise and Salt-and-Pepper impulse noise).
- Compare linear spatial convolution (Gaussian Blur) against non-linear order-statistic filtering (Median Blur) and evaluate quantitative restoration quality using **PSNR (dB)** and **SSIM**.
- Apply and contrast classical spatial segmentation strategies: Global Fixed, Otsu's inter-class variance maximization, Adaptive Gaussian thresholding, Mathematical Morphology, Canny edge detection, and Marker-Controlled Watershed segmentation.
- Extract dense gradient descriptors (**HOG**) and sparse interest points (**SIFT**), evaluating transformation invariance across controlled rotation ($45^\circ$) and scaling ($0.5\times$).
- Contrast classical sliding-window feature cascades (**Viola-Jones Haar Cascade**) against modern single-stage deep convolutional networks (**Ultralytics YOLOv8**).
- Analyze temporal dynamics across streaming video using **Frame Differencing**, **Gaussian Mixture Model Background Subtraction (MOG2)**, and **Sparse Lucas-Kanade Optical Flow**.

---

## Summary of the Four Tasks

### Task 1: Image Restoration and Segmentation
- **Image Inspection**: Reads raw BGR image via OpenCV, converts to RGB, reports dimensions, channel depth, and intensity range.
- **Noise Injection**:
  - *Gaussian Noise:* Simulates thermal electronic noise $\eta \sim \mathcal{N}(0, \sigma^2)$ with dynamic range clipping $[0, 255]$.
  - *Salt-and-Pepper Noise:* Simulates sensor dropouts and bit-flip errors via random coordinate assignment of black ($0$) and white ($255$) pixels.
- **Spatial Filtering**:
  - Gaussian linear smoothing ($5\times 5$, $\sigma=1.5$).
  - Median rank-order filter ($5\times 5$).
- **Quantitative Quality Assessment**:
  - Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) against pristine reference image.
  - Automatically exports tabular metrics to `outputs/metrics/task1_restoration_metrics.csv`.
- **Segmentation & Boundary Analysis**:
  - Global Binary Thresholding ($T=127$).
  - Otsu's Optimal Automatic Thresholding.
  - Adaptive Gaussian Thresholding ($11\times 11$, $C=2$).
  - Mathematical Morphology: Erosion, Dilation, Opening, Closing ($5\times 5$ structuring element).
  - Multi-stage Canny Edge Detection ($T_{low}=50, T_{high}=150$).
  - Marker-controlled Watershed Segmentation using Euclidean distance transform local maxima.

### Task 2: Feature Extraction and Matching
- **Histogram of Oriented Gradients (HOG)**:
  - 9 orientation bins, $8\times 8$ pixels per cell, $2\times 2$ cells per block, L2-Hys block normalization.
  - Produces high-dimensional vector capturing local edge structure and shape contours.
- **Scale-Invariant Feature Transform (SIFT)**:
  - Difference-of-Gaussians (DoG) scale-space extrema detection.
  - Hessian-based unstable edge elimination and sub-pixel localization.
  - 128-dimensional orientation descriptors invariant to scaling and rotation.
- **Correspondence Matching**:
  - Brute-Force Matcher with Euclidean distance ($L_2$).
  - Lowe's Second-Nearest-Neighbor Ratio Test ($d_1 / d_2 < 0.75$) to filter ambiguous candidate matches.
- **Empirical Invariance Benchmarks**:
  - *Experiment 1:* Original vs. $45^\circ$ Affine-rotated image.
  - *Experiment 2:* Original vs. $0.5\times$ Rescaled image.
  - *Experiment 3:* Original vs. Secondary supplied image.
  - Exports retention rate and match counts to `outputs/metrics/task2_feature_metrics.csv`.

### Task 3: Object Recognition
- **Classical Detector (Viola-Jones / Haar Cascade)**:
  - Evaluates rectangular features using Integral Images in $O(1)$ constant time.
  - Applies OpenCV pre-trained frontal face cascade (`haarcascade_frontalface_default.xml`).
  - Benchmarks execution time and isolates detection coordinates.
- **Deep Learning Detector (Ultralytics YOLOv8)**:
  - Single-stage end-to-end convolutional neural network (`yolov8n.pt`).
  - Detects objects across 80 MS COCO categories simultaneously.
  - Records class predictions, confidence scores, and bounding boxes in `outputs/task3_object_recognition/yolo_detection_records.csv`.
- **Architectural Comparison**:
  - Benchmarks detection targets, latency (ms), throughput (FPS), hardware profiles, and failure modes.
  - Exports comparative summary to `outputs/metrics/task3_detection_metrics.csv`.

### Task 4: Video Motion Estimation
- **Stream Verification**: Decodes video stream frame-by-frame, extracting resolution, frame count, FPS, and sample frames.
- **Temporal Frame Differencing**:
  - Computes $|I_t - I_{t-1}|$ followed by thresholding ($T=25$) and morphological opening.
  - Ultra-high throughput ($O(N)$), ideal for static surveillance with rapid triggers.
- **GMM Background Subtraction (MOG2)**:
  - Dynamically models multi-modal pixel color distributions ($K=3\dots 5$).
  - Explicit chromatic shadow suppression (distinguishing shadows from true moving bodies).
  - Extracts contours and draws bounding boxes around moving foreground objects.
- **Sparse Lucas-Kanade Optical Flow**:
  - Tracks salient Shi-Tomasi corners across Gaussian pyramids (`calcOpticalFlowPyrLK`).
  - Computes velocity displacement vectors and visualizes real-time motion trajectory trails.
- **Method Comparison**:
  - Compares mathematical principles, throughput (FPS), occlusion handling, and camera motion tolerance.
  - Exports benchmarks to `outputs/metrics/task4_motion_metrics.csv`.

---

## Technologies and Libraries Used
- **Python 3.10+** (tested on Python 3.10, 3.11, 3.12, 3.13)
- **NumPy** (`>=1.24.0`): Matrix operations and array noise synthesis.
- **OpenCV (`opencv-python`)** (`>=4.8.0`): Core imaging, spatial filters, SIFT, Haar cascades, video decoding, and optical flow.
- **scikit-image** (`>=0.21.0`): Perceptual SSIM metric computation and HOG feature generation.
- **Pandas** (`>=2.0.0`): Tabular metric synthesis and CSV persistence.
- **Matplotlib** (`>=3.7.0`): Publication-grade multi-panel visual rendering.
- **Ultralytics YOLO** (`>=8.0.0`): Modern deep learning object detection with YOLOv8.
- **SciPy** (`>=1.10.0`): Multidimensional image processing routines.
- **Jupyter / Google Colab**: Interactive notebook execution environment.

---

## Project Directory Structure

```text
CCA-3-Computer-Vision/
├── CCA_3_Computer_Vision_Assignment.ipynb    # Main Colab notebook (All 4 tasks)
├── test_cv_algorithms.py                     # Automated algorithmic & structure test suite
├── README.md                                 # Complete project documentation
├── requirements.txt                          # Project dependency specifications
├── .gitignore                                # Git ignore file for artifacts/data
│
├── data/
│   ├── images/                               # Place your 1 or 2 images here
│   │   └── .gitkeep
│   └── videos/                               # Place your 1 video here
│       └── .gitkeep
│
├── outputs/
│   ├── task1_restoration/                    # Restored, noisy, and filtered images
│   ├── task1_segmentation/                   # Thresholded, morphological, edge & watershed images
│   ├── task2_features/                       # HOG, SIFT, and invariance matching figures
│   ├── task3_object_recognition/             # Haar and YOLO annotated images & detection CSV
│   ├── task4_motion_estimation/              # Differencing, MOG2, and optical flow frames
│   └── metrics/                              # Real CSV metrics from all 4 tasks
│
└── reports/
    └── CCA_3_Computer_Vision_Report.md        # Academic report template & methodology
```

### File and Directory Descriptions

| File / Folder Path | Type | Purpose & Details |
| :--- | :--- | :--- |
| `CCA_3_Computer_Vision_Assignment.ipynb` | Jupyter Notebook | Complete self-contained notebook implementing all 4 core tasks: Restoration & Segmentation, Feature Extraction & SIFT Matching, Haar & YOLOv8 Object Recognition, and Video Motion Estimation. |
| `test_cv_algorithms.py` | Python Script | Automated test suite that validates project structure, notebook JSON schema, mathematical formulas (PSNR, MSE), noise generation, morphological filters, and missing file error handling. |
| `requirements.txt` | Configuration | Pinned project dependencies ensuring seamless reproducibility across local Python virtual environments and Google Colab runtimes. |
| `README.md` | Markdown Documentation | Comprehensive project overview, technical breakdown, setup instructions, and evaluation guidelines. |
| `.gitignore` | Configuration | Specifies untracked files and directories (virtual environments, cache folders, local models, and temporary files). |
| `data/images/` | Input Directory | Directory for user-supplied test images (Primary image for Tasks 1-3, Secondary image for Task 2 matching). |
| `data/videos/` | Input Directory | Directory for user-supplied video footage used for Task 4 motion estimation. |
| `outputs/` | Output Directory | Root directory partitioned into task-specific subfolders for saved visual panels, segmentation masks, feature matches, detection records, and quantitative CSV metrics. |
| `reports/` | Academic Report | Contains `CCA_3_Computer_Vision_Report.md` providing in-depth theoretical analysis, mathematical formulations, and experimental evaluations. |

---

## Supplying Your Own Input Data
The project supports providing your own personal images and videos without modifying the underlying algorithm cells.

### Local Method: Data Directory
1. Place 1 or 2 images (`.jpg`, `.jpeg`, `.png`, `.bmp`) in:
   ```text
   data/images/
   ```
   - Primary Image: Used for Task 1, Task 2, and Task 3.
   - Secondary Image: Used for Task 2 cross-image matching.
2. Place 1 video (`.mp4`, `.avi`, `.mov`) in:
   ```text
   data/videos/
   ```
   - Target Video: Used for Task 4 motion estimation.

### Google Colab Direct Upload
Run **Cell 3.1** in the notebook. An interactive file selector will prompt you to upload files directly from your computer. The notebook automatically routes images to `data/images/` and videos to `data/videos/`.

> **Missing File Safeguard:** If no files are present, the notebook displays a clear notification explaining which files are needed and allows theoretical and environment cells to run without crashing.

---

## Installation & Setup Instructions

### Local Windows Execution

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "C:\path\to\CCA_3_Computer_Vision"
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Upgrade pip and install project dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```powershell
   jupyter notebook CCA_3_Computer_Vision_Assignment.ipynb
   ```

5. In the Jupyter interface, execute the cells sequentially from top to bottom.

---

### Google Colab Execution

1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `CCA_3_Computer_Vision_Assignment.ipynb`.
3. In the first code cell, uncomment and run the dependency installation line:
   ```python
   !pip install -q opencv-python scikit-image ultralytics pandas matplotlib scipy
   ```
4. Run **Cell 3.1** to upload your personal images and video.
5. Select **Runtime $\to$ Run all** or step through each section.
6. In **Section 9**, run the inventory cell to inspect and verify all generated output files and metric tables.

---

## Explanation of Output Directories

| Directory Path | Contents Description |
| :--- | :--- |
| `outputs/task1_restoration/` | Pristine image, Gaussian noisy image, Salt-and-Pepper noisy image, Gaussian-filtered outputs, Median-filtered outputs, and 4-panel comparison figures. |
| `outputs/task1_segmentation/` | Global binary, Otsu's binary, Adaptive Gaussian thresholding, morphological operation panels (Erosion, Dilation, Opening, Closing), Canny edge maps, and Marker-controlled Watershed segmentation overlays. |
| `outputs/task2_features/` | HOG gradient field visualization, SIFT rich keypoint displays, two-image correspondence matching lines, $45^\circ$ rotation invariance panels, and $0.5\times$ scale invariance panels. |
| `outputs/task3_object_recognition/`| Haar Cascade facial bounding box annotations, Ultralytics YOLOv8 multi-class detection visualizations, and `yolo_detection_records.csv`. |
| `outputs/task4_motion_estimation/` | Video inspection sample frames, Temporal Frame Differencing motion masks, MOG2 foreground masks and tracking boxes, and Lucas-Kanade optical flow vector tracks. |
| `outputs/metrics/` | Programmatically generated CSV files: `task1_restoration_metrics.csv`, `task2_feature_metrics.csv`, `task3_detection_metrics.csv`, and `task4_motion_metrics.csv`. |

---

## Running Tests and Algorithmic Validation
To verify mathematical routines and ensure all functions operate reliably without errors on test matrices, a dedicated test script (`test_cv_algorithms.py`) is provided.

Run the test suite:
```powershell
python test_cv_algorithms.py
```
Or using pytest:
```powershell
pytest -v test_cv_algorithms.py
```

### What the test suite validates:
1. **Gaussian Noise Generator:** Confirms output preserves dimensions, valid dynamic range $[0, 255]$, and expected variance shift.
2. **Salt-and-Pepper Generator:** Confirms correct pixel corruption ratio and exact impulse values ($0$ and $255$).
3. **PSNR Calculation:** Confirms $PSNR = \infty$ for identical images and expected dB drop for degraded images.
4. **SSIM Calculation:** Confirms $SSIM = 1.0$ for identical images and $SSIM \in [0, 1]$ for perturbed images.
5. **Morphological Filters:** Confirms mathematical set properties ($A \ominus B \subseteq A \subseteq A \oplus B$).
6. **Canny Edge Detection:** Confirms binary edge output with zero values in homogenous areas.
7. **Lucas-Kanade Functionality:** Confirms Shi-Tomasi corner detection and optical flow structure.
8. **Notebook Integrity:** Confirms valid JSON syntax and checks cell presence across all task sections.

*(Note: Synthetic tests verify software correctness only; actual experimental results are computed when custom media files are loaded).*

---

## Project Limitations
- **Thresholding in Non-uniform Lighting:** Global and Otsu's thresholding methods assume bimodal histograms and degrade under steep illumination gradients.
- **Aperture Effect in Optical Flow:** Lucas-Kanade cannot resolve motion along homogenous, textureless edges where gradients are one-dimensional.
- **Ghosting in Frame Differencing:** Temporal subtraction $|I_t - I_{t-1}|$ generates both leading and trailing silhouettes and fails when objects become stationary.
- **Deep Learning Resource Demands:** YOLO achieves superior semantic detection but requires greater GPU/CPU memory bandwidth than classical Haar cascades.

---

## Academic Integrity Statement
This codebase is prepared strictly for academic coursework submission and project evaluation. All code implementations, mathematical derivations, and technical explanations have been authored from first principles. When experimental data is supplied, all metrics and visualizations are derived from direct execution on custom data without synthetic fabrication.
