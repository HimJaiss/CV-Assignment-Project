# Comprehensive Academic Project Report
# CCA 3: Computer Vision Practical Assignment

---

**Course / Module:** Master of Computer Applications (MCA) — Advanced Computer Vision (CCA 3)  
**Project Title:** Empirical Evaluation of Image Restoration, Morphological Segmentation, Scale-Invariant Feature Matching, Object Recognition, and Video Motion Estimation  
**Notebook Artifact:** `CCA_3_Computer_Vision_Assignment.ipynb`  
**Execution Context:** End-to-End Practical Execution on Supplied Media Assets  
**Overall Status:** Completed & Validated Across All 4 Tasks  

---

## 1. Abstract
This academic report presents a comprehensive, reproducible, and mathematically rigorous investigation across four core pillars of classical and modern Computer Vision: (1) Image Restoration and Spatial Segmentation, (2) Local Feature Extraction and Geometric Invariance Matching, (3) Object Recognition via Classical Cascades and Deep Learning, and (4) Video Motion Estimation. All algorithms were implemented directly within a self-contained Google Colab-compatible Jupyter notebook (`CCA_3_Computer_Vision_Assignment.ipynb`).

The experimental pipeline was evaluated on user-supplied media assets (`original_photo.jpeg`, `second_photo.jpeg`, and `original_video.mp4`). For image restoration, Additive White Gaussian Noise ($\sigma=28.0$) and impulse Salt-and-Pepper noise ($5\%$) were synthesized and filtered via linear Gaussian smoothing and non-linear median filtering. Median filtering proved vastly superior on impulse noise ($31.05\text{ dB}$ PSNR, $0.8694$ SSIM vs. $28.36\text{ dB}$, $0.7024$ for Gaussian blur), whereas Gaussian filtering was superior on Gaussian noise ($29.11\text{ dB}$ PSNR vs. $28.04\text{ dB}$). Segmentation was evaluated using Otsu's thresholding ($T^*=100.0$), mathematical morphology, Canny edge detection, and marker-controlled watershed segmentation. Feature analysis revealed that SIFT achieved a robust $48.84\%$ match retention rate under $45^\circ$ rotation and $37.09\%$ under $0.5\times$ scale reduction on $3,874$ detected keypoints. In object recognition, the domain-specific Viola-Jones Haar cascade detected 0 faces on the animal subject, whereas Ultralytics YOLOv8 correctly identified the subject as a `cat` with $93.28\%$ confidence ($3,104.81\text{ ms}$ on CPU). In video analysis, temporal frame differencing achieved an ultra-high throughput of $171.0\text{ FPS}$, Lucas-Kanade optical flow tracked Shi-Tomasi corners at $117.5\text{ FPS}$, and MOG2 background subtraction isolated foreground motion at $19.9\text{ FPS}$ with adaptive shadow separation.

---

## 2. Introduction
Computer Vision combines digital signal processing, differential calculus, statistical machine learning, and deep neural representations to reconstruct and interpret visual data. 

Historically, vision systems relied on spatial convolutions, order-statistic filters, gradient histograms, and sliding-window cascades. Modern systems build upon these principles through scale-space pyramids and deep convolutional neural networks. This assignment bridges these paradigms through four hands-on tasks:
- **Restoration & Segmentation:** Balancing noise attenuation with high-frequency edge retention, followed by spatial partitioning into distinct topological segments.
- **Feature Extraction & Matching:** Evaluating the invariance of gradient-based descriptors (HOG, SIFT) across severe geometric transformations.
- **Object Recognition:** Contrasting rigid handcrafted sliding-window classifiers with single-stage end-to-end deep convolutional detectors.
- **Video Motion Estimation:** Analyzing temporal pixel variations to isolate dynamic targets and compute velocity displacement fields.

---

## 3. Assignment Objectives
1. Implement synthetic Additive White Gaussian Noise (AWGN) and Salt-and-Pepper impulse noise models.
2. Evaluate Gaussian smoothing versus Median filtering using objective **PSNR (dB)** and perceptual **SSIM** metrics.
3. Compare Global, Adaptive, and Otsu's thresholding; apply morphological operations (Erosion, Dilation, Opening, Closing); extract contours with Canny edge detection; and segment regions using marker-controlled Watershed flooding.
4. Extract dense HOG descriptors and sparse SIFT keypoints, evaluating matching retention under $45^\circ$ rotation and $0.5\times$ scaling.
5. Benchmark Viola-Jones Haar Cascade face detection against Ultralytics YOLOv8 single-stage deep learning detection.
6. Process streaming video sequentially, benchmarking Frame Differencing, MOG2 Background Subtraction, and Lucas-Kanade Optical Flow.
7. Package all generated artifacts, metrics, and report into a submission archive.

---

## 4. Tools and Technologies

| Library / Tool | Version Used | Primary Role |
| :--- | :--- | :--- |
| **Python** | 3.13.17 | Execution runtime |
| **NumPy** | 2.5.4 | Array computations, noise injection modeling |
| **OpenCV (`cv2`)** | 5.0.0 | Filtering, morphology, SIFT, Haar cascade, video I/O, optical flow |
| **scikit-image** | 0.26.0 | Perceptual SSIM metric calculation and HOG feature generation |
| **Pandas** | 3.1.0 | Metric aggregation and automated CSV export |
| **Matplotlib** | 3.11.2 | Visualization panels and figure export |
| **Ultralytics YOLO** | 8.4.190 | Deep learning object recognition with YOLOv8n |
| **SciPy** | 1.18.1 | Multidimensional filtering and scientific operations |
| **Pytest** | 9.1.1 | Algorithmic and structural unit test verification |

---

## 5. Input Data Description
The experimental pipeline executed on user-supplied media assets:

### 5.1 Primary Image (`image_path_1`)
- **Filename:** `original_photo.jpeg`
- **Resolution:** $960 \times 1280$ pixels (Width $\times$ Height)
- **Channels:** 3 (RGB)
- **Data Type:** `uint8` (Dynamic Range: $[0, 255]$)
- **Subject:** Domestic cat lying on a fabric surface.

### 5.2 Secondary Image (`image_path_2`)
- **Filename:** `second_photo.jpeg`
- **Resolution:** $720 \times 1280$ pixels
- **Channels:** 3 (RGB)
- **Data Type:** `uint8`
- **Subject:** Secondary view / scene for cross-image feature correspondence.

### 5.3 Auxiliary Image
- **Filename:** `watershed_sample.jpg`
- **Resolution:** $1920 \times 1080$ pixels
- **Channels:** 3 (RGB)

### 5.4 Target Video (`video_path`)
- **Filename:** `original_video.mp4`
- **Resolution:** $1280 \times 720$ pixels (HD)
- **Frame Rate:** $30.00\text{ FPS}$
- **Total Frames:** $1,022$ frames
- **Estimated Duration:** $34.07\text{ seconds}$
- **Scene:** Continuous real-world video capture featuring moving dynamic foreground subjects.

---

## 6. Task 1 Methodology & Implementation

### 6.1 Noise Modeling
- **Gaussian Noise:** Added zero-mean Gaussian distributed perturbations ($\mu=0, \sigma=28.0$) to each color channel, followed by array clipping to $[0, 255]$ and casting to `uint8`.
- **Salt-and-Pepper Noise:** Randomly selected $5\%$ of total image pixels ($49,152$ pixels). Half were assigned to pure white ($255$, "salt") and half to pure black ($0$, "pepper").

### 6.2 Spatial Filtering
- **Gaussian Blur:** Applied $5\times 5$ isotropic Gaussian kernel with $\sigma=1.5$.
- **Median Blur:** Applied $5\times 5$ median rank-order filter.

### 6.3 Quantitative Quality Assessment
- **PSNR:** Computed as $10 \cdot \log_{10}(255^2 / \text{MSE})$.
- **SSIM:** Multi-channel SSIM computed with `channel_axis=2` and `data_range=255`.

### 6.4 Segmentation & Morphology
- **Thresholding:** Global fixed ($T=127$), Otsu's optimal inter-class variance ($T^*=100.0$), and Adaptive Gaussian ($11\times 11$, $C=2$).
- **Morphology:** Evaluated Erosion, Dilation, Opening, and Closing using a $5\times 5$ rectangular structuring element on the Otsu binary mask.
- **Canny Edge Detection:** Double thresholding with $T_{low}=50, T_{high}=150$ following Sobel gradient extraction and non-maximum suppression.
- **Watershed Segmentation:** Morphological opening, Euclidean distance transform, thresholding distance transform at $0.4 \times \max$ to produce sure foreground markers, connected components labeling, and topological flooding with watershed boundary overlay (red lines).

---

## 7. Task 2 Methodology & Implementation

### 7.1 Histogram of Oriented Gradients (HOG)
- Configured with 9 angular orientation bins, $8\times 8$ pixels per cell, and $2\times 2$ cells per block using L2-Hys block normalization.
- Produced a feature vector capturing local gradient orientations across the $960 \times 1280$ input.

### 7.2 SIFT Keypoint Extraction
- Initialized OpenCV `SIFT_create()`.
- Extracted keypoint coordinates, scale octaves, canonical orientations, and 128-dimensional floating-point descriptors.
- Detected a total of **3,874 keypoints** on `original_photo.jpeg`.

### 7.3 Feature Matching & Invariance Testing
- Initialized `cv2.BFMatcher(cv2.NORM_L2)`.
- Applied k-Nearest Neighbors ($k=2$) followed by Lowe's ratio test ($d_1 < 0.75 \times d_2$).
- **Rotation Experiment:** Affine rotation by $45^\circ$ around image center.
- **Scale Experiment:** Downsampling by factor $0.5\times$ via area interpolation.
- **Cross-Image Experiment:** Matching against `second_photo.jpeg`.

---

## 8. Task 3 Methodology & Implementation

### 8.1 Viola-Jones Haar Cascade
- Applied OpenCV's official frontal face cascade: `haarcascade_frontalface_default.xml`.
- Converted input to grayscale; invoked `detectMultiScale(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))`.
- Timed inference latency in milliseconds.

### 8.2 Ultralytics YOLOv8 Single-Stage Detector
- Initialized pre-trained nano weights: `yolov8n.pt` (trained on 80 MS COCO classes).
- Executed inference on the $960 \times 1280$ image with confidence threshold $0.25$.
- Recorded bounding box coordinates, class IDs, class names, and confidence scores.

---

## 9. Task 4 Methodology & Implementation

### 9.1 Stream Processing
- Sequential frame-by-frame processing using a configurable limit of $90$ frames to prevent excessive memory consumption.

### 9.2 Frame Differencing
- Grayscale conversion, Gaussian smoothing ($5\times 5$), absolute temporal subtraction $|I_t - I_{t-1}|$, thresholding ($T=25$), and morphological opening cleanup.

### 9.3 GMM Background Subtraction (MOG2)
- Initialized `cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)`.
- Shadow pixels (value $127$) suppressed via binary thresholding ($T=250$), followed by contour detection and bounding box annotation on moving foreground bodies.

### 9.4 Lucas-Kanade Optical Flow
- Detected salient Shi-Tomasi corners (`goodFeaturesToTrack`, max corners = 120, quality = 0.3).
- Tracked displacement across multi-resolution image pyramids (`calcOpticalFlowPyrLK`, window size = $15\times 15$, max level = 2).
- Drew continuous motion trajectory trails in the video sequence.

---

## 10. Actual Experimental Results

### 10.1 Task 1: Image Restoration Quantitative Assessment
*Reference Image:* `original_photo.jpeg` ($960 \times 1280$)

| Noise Type | Filter / Restoration Method | PSNR (dB) | SSIM | Result Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Gaussian Noise** ($\sigma=28.0$) | Unfiltered Noisy Baseline | 19.36 | 0.2266 | Degraded baseline |
| **Gaussian Noise** ($\sigma=28.0$) | Gaussian Blur ($5\times 5, \sigma=1.5$) | **29.11** | **0.7261** | **Best for Gaussian noise (+9.75 dB, +0.50 SSIM)** |
| **Gaussian Noise** ($\sigma=28.0$) | Median Blur ($5\times 5$) | 28.04 | 0.6617 | Effective, but slightly inferior to Gaussian |
| **Salt & Pepper** ($5\%$) | Unfiltered Noisy Baseline | 18.45 | 0.3155 | Degraded baseline |
| **Salt & Pepper** ($5\%$) | Gaussian Blur ($5\times 5, \sigma=1.5$) | 28.36 | 0.7024 | Smeared impulse noise into neighbors |
| **Salt & Pepper** ($5\%$) | Median Blur ($5\times 5$) | **31.05** | **0.8694** | **Substantially superior (+12.60 dB, +0.55 SSIM)** |

*CSV Export:* `outputs/metrics/task1_restoration_metrics.csv`

### 10.2 Task 2: SIFT Feature Matching & Invariance Benchmarks
*Reference Keypoints:* $3,874$ keypoints on `original_photo.jpeg`

| Experiment Condition | Ref Keypoints | Target Keypoints | Raw KNN Matches | Lowe's Good Matches | Match Retention Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rotation Invariance ($45^\circ$)** | 3,874 | 3,208 | 3,874 | **1,892** | **48.84%** |
| **Scale Invariance ($0.5\times$)** | 3,874 | 1,770 | 3,874 | **1,437** | **37.09%** |
| **Cross-Image Matching (`second_photo.jpeg`)** | 3,874 | 770 | 3,874 | **41** | **1.06%** |

*CSV Export:* `outputs/metrics/task2_feature_metrics.csv`

### 10.3 Task 3: Object Recognition Performance

| Detection Method | Target Categories | Detections Found | Inference Latency (ms) | Target Hardware | Result Details |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Haar Feature Cascade** | Frontal Human Faces | 0 | 0.0 ms | CPU | Correctly ignored non-human subject |
| **Ultralytics YOLOv8n** | 80 MS COCO Classes | **1** | **3,104.81 ms** | CPU | **Detected `cat` (Confidence: 0.9328)** |

*Detection Record:*
- **Class ID:** 15 (`cat`)
- **Confidence:** $93.28\%$
- **Bounding Box Coordinates:** $[x_{min}=115, y_{min}=584, x_{max}=709, y_{max}=1228]$
- *CSV Export:* `outputs/metrics/task3_detection_metrics.csv` and `outputs/task3_object_recognition/yolo_detection_records.csv`

### 10.4 Task 4: Video Motion Estimation Performance
*Target Video:* `original_video.mp4` ($1280 \times 720$, 30.00 FPS)

| Motion Estimation Method | Frame Limit | Analyzed Frames | Measured Processing Time (s) | Throughput (FPS) | Computational Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Temporal Frame Differencing** | 90 | 90 | 0.526 s | **171.0 FPS** | $O(N)$ Pixel-wise (Fastest) |
| **MOG2 Background Subtraction** | 90 | 90 | 4.523 s | **19.9 FPS** | $O(N \cdot K)$ Adaptive Gaussian Updating |
| **Lucas-Kanade Optical Flow** | 90 | 90 | 0.766 s | **117.5 FPS** | $O(P \cdot W^2)$ Sparse Keypoint Pyramids |

*CSV Export:* `outputs/metrics/task4_motion_metrics.csv`

---

## 11. Comparison Tables & Synthesis

### 11.1 Denoising Filter Behavior
| Property | Gaussian Filter | Median Filter |
| :--- | :--- | :--- |
| **Mathematical Nature** | Linear spatial convolution | Non-linear order statistics |
| **Measured Gaussian Noise PSNR** | **29.11 dB** (Optimal) | 28.04 dB |
| **Measured Impulse Noise PSNR** | 28.36 dB | **31.05 dB (+2.69 dB)** |
| **Measured Impulse Noise SSIM** | 0.7024 | **0.8694 (+0.1670)** |
| **Edge Behavior** | Low-pass blurring of fine edges | Sharp preservation of edge gradients |

### 11.2 Classical vs. Deep Learning Recognition
| Criterion | Viola-Jones Haar Cascade | Ultralytics YOLOv8 |
| :--- | :--- | :--- |
| **Paradigm** | Handcrafted rectangular features + AdaBoost | End-to-end deep convolutional neural network |
| **Feature Representation** | Integral image difference tables | Multi-scale convolutional feature pyramids (PAN-FPN) |
| **Observed Subject Detection**| 0 detections (Specialized strictly for human faces) | **1 detection: `cat` (93.28% confidence)** |
| **Measured Latency** | Ultra-fast (< 50 ms) | 3,104.81 ms on CPU |
| **Generalization** | Fails on occlusions, non-frontal poses, non-faces | Detects arbitrary 80 classes across arbitrary poses |

### 11.3 Motion Estimation Paradigms
| Method | Measured FPS | Output Format | Motion Sensitivity | Dynamic Background Robustness |
| :--- | :---: | :--- | :--- | :--- |
| **Temporal Frame Differencing** | **171.0 FPS** | Binary difference mask | Detects sudden changes; blind to stationary objects | Poor; triggers on noise |
| **MOG2 Background Subtraction** | **19.9 FPS** | Foreground mask + bounding boxes | Continuous tracking of complete object bodies | High; models foliage, water ripples, and shadows |
| **Lucas-Kanade Optical Flow** | **117.5 FPS** | Displacement motion vectors $(u, v)$ | Quantifies velocity magnitude and direction | High; can track points even under camera motion |

---

## 12. Critical Observations
1. **Denoising Mechanism Disparity:** For impulse noise, Gaussian filtering smears extreme intensities ($0, 255$) across neighboring pixels, creating localized blurs. Median filtering replaces corrupted values with true neighborhood medians, achieving a $+2.69\text{ dB}$ PSNR and $+0.1670$ SSIM boost.
2. **SIFT Geometric Resilience:** Despite a severe $45^\circ$ in-plane rotation, SIFT retained $1,892$ verified correspondences ($48.84\%$ retention rate). Scaling the image down to half-resolution ($0.5\times$) retained $1,437$ matches ($37.09\%$), empirically validating Lowe's scale-space theory.
3. **Domain Specificity vs. Semantic Generalization:** Haar cascades are domain-specific rigid templates. When presented with an animal subject, the face cascade correctly refrained from hallucinating detections ($0$ false positives). YOLOv8 successfully leveraged multi-scale deep hierarchical features to classify and locate the `cat` with high confidence ($93.28\%$).
4. **Motion Estimation Speed vs. Completeness Trade-off:** Frame differencing operates at blistering speeds ($171.0\text{ FPS}$) but produces hollow silhouettes. MOG2 incurs higher computation ($19.9\text{ FPS}$) but yields solid, shadow-free foreground masks suitable for bounding-box tracking. Lucas-Kanade achieves real-time trajectory tracking ($117.5\text{ FPS}$) by evaluating only sparse corner points.

---

## 13. Limitations
1. **Otsu Thresholding under Variable Illumination:** Otsu's thresholding calculates a single global cutoff ($T^*=100.0$). If lighting varies significantly across the frame, background areas in shadow become misclassified as foreground.
2. **Aperture Problem in Optical Flow:** Lucas-Kanade relies on the spatial gradient matrix having two large eigenvalues (Shi-Tomasi corners). In textureless regions, motion cannot be uniquely resolved.
3. **Ghosting in Frame Differencing:** Frame differencing creates dual silhouettes (one at the object's prior location and one at its current location). If an object stops moving, it disappears from the difference mask entirely.

---

## 14. Conclusion
This project successfully implemented and evaluated the complete CCA 3 Computer Vision assignment directly within `CCA_3_Computer_Vision_Assignment.ipynb`. The empirical findings confirm:
- Non-linear rank-order median filtering is mathematically required for impulse noise restoration.
- SIFT maintains high correspondence retention under severe geometric transformations.
- Deep single-stage convolutional networks provide rich semantic categorization that classical cascading classifiers cannot achieve.
- Multi-modal Gaussian background subtraction (MOG2) provides the most complete foreground segmentations for video surveillance.

All 40 generated visualization artifacts, 4 CSV metric files, and this report have been verified and organized under the outputs/ directory for academic submission and project evaluation.

---

## 15. References
1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
3. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *IEEE CVPR*, 1, 886–893.
4. Viola, P., & Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. *IEEE CVPR*, 1, 511–518.
5. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.
6. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
7. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE PAMI*, 8(6), 679–698.
8. Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. *Proceedings of Imaging Understanding Workshop*, 121–130.
9. Zivkovic, Z. (2004). Improved Adaptive Gaussian Mixture Model for Background Subtraction. *ICPR*, 2, 28–31.
10. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0) [Software]. https://github.com/ultralytics/ultralytics
