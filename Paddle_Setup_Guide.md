# **Setup Guide: PaddleOCR Video Annotation Pipeline**

A complete, production-ready guide for running a **GPU-accelerated PaddleOCR pipeline** that extracts frames from a video, detects regions using OCR models, and exports annotations compatible with **X-AnyLabeling**.

---

# **1. Hardware & System Requirements**

## **Minimum Hardware (Works but Slow):**

| Component   | Requirement                                                 |
| ----------- | ----------------------------------------------------------- |
| **GPU**     | NVIDIA GPU with **6GB+ VRAM** (GTX 1660 Ti, RTX 2060, etc.) |
| **CPU**     | Any 4-core processor                                        |
| **RAM**     | 16GB                                                        |
| **Storage** | 10GB free space                                             |
| **OS**      | Linux / WSL2 / Windows with Docker Desktop                  |

## **Recommended Hardware (Smooth):**

| Component   | Requirement                            |
| ----------- | -------------------------------------- |
| **GPU**     | NVIDIA **RTX 3060, 4060, A5000, etc.** |
| **RAM**     | 32GB                                   |
| **Storage** | SSD with 20GB+                         |

---

# **2. Software & OS Requirements**

### **Required Software**

* Docker **with NVIDIA GPU support**
* NVIDIA drivers installed (**525+ recommended**)
* WSL2 (for Windows users)
* CUDA runtime inside Docker image (already provided)
* Python inside Docker (installed automatically by image)

### **Verify GPU Driver**

```bash
nvidia-smi
```

➤ If this command fails, fix drivers before continuing.

---

# **3. Quick Setup (TL;DR)**

## **Step 1 — Pull PaddleOCR Docker Image**

CUDA 11.8 image:

```bash
docker pull paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9
```

CUDA 12.6 image:

```bash
docker pull registry.baidubce.com/paddlepaddle/paddle:3.2.2-gpu-cuda12.6-cudnn9.5-trt10.5
```

---

# **4. Installing NVIDIA Container Toolkit**

## **For Windows (WSL2)**

### **1. Install WSL (if not installed)**

```bash
wsl --install
```

### **2. Ensure WSL uses version 2**

```bash
wsl --set-default-version 2
```

### **3. Install Docker Desktop**

Enable:
✔ *Use WSL 2 based engine*
✔ *Enable GPU support*

### **4. Restart machine**

---

## **For Linux**

### **1. Add NVIDIA package repo**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### **2. Install toolkit**

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### **3. Enable runtime**

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### **4. Restart Docker**

```bash
sudo systemctl restart docker
```

### **5. Validate**

```bash
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

# **5. Running the Docker Container**

### **Correct command (WSL + Linux)**

```bash
docker run --gpus all -it -v "$(pwd)":/workspace paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9 bash
```

### **Common Fixes**

❌ **“invalid reference format”**
Occurs if using PowerShell with `"$(pwd)"`.

✅ Use:

```powershell
docker run --gpus all -it -v "${pwd}:/workspace" ...
```

---

# **6. Installing Python Dependencies (Inside Container)**

```bash
pip install paddleocr paddlepaddle-gpu opencv-python-headless numpy
pip install "paddlex[ocr]"
```

### **Check installation**

```bash
python -c "import paddleocr; print('OK')"
```

---

# **7. Create Project Structure**

Inside the container:

```bash
cd /workspace
mkdir -p data output_video logs
```

Place your videos in:

```
/workspace/data/
```

---

# **8. Running the Video Annotation Pipeline**

### **Command**

```bash
python region_detection.py --video your_video.mp4 --fps 1
```

### **Parameter Explanation**

| Argument       | Meaning                            |
| -------------- | ---------------------------------- |
| `--video / -v` | Name of video inside `/data/`      |
| `--fps / -f`   | How many frames to extract per sec |
| `1 fps`        | 1 frame every second               |
| `0.5 fps`      | 1 frame every 2 seconds            |
| `5 fps`        | Extract more detail (slower)       |

---

# **9. Output Directory Structure (Complete)**

```
output_video/
└── your_video/
    ├── annotated_video.mp4      # Rendered result
    ├── video_annotations.json   # Combined output
    ├── classes.txt              # Auto-detected class labels
    ├── export/                  # X-AnyLabeling compatible
    │   ├── frame_0001.jpg
    │   ├── frame_0001.json
    │   ├── frame_0002.jpg
    │   ├── frame_0002.json
    │   └── ...
    └── raw_frames/              # Optional dump of all frames
```

---

# **10. Using X-AnyLabeling**

### **1. Download**

[https://github.com/CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
## 1. Installation and Deployment

X-AnyLabeling provides multiple installation methods. You can install the official package directly via `pip` to get the latest stable version, install from source by cloning the official GitHub repository, or use the convenient GUI installer package.

> [!NOTE]
> **Advanced Features**: The following advanced features are only available through Git clone installation. Please refer to the corresponding documentation for configuration instructions.
>
> 0. **Remote Inference Service**: X-AnyLabeling-Server based remote inference service - [Installation Guide](https://github.com/CVHub520/X-AnyLabeling-Server)
> 1. **Video Object Tracking**: Segment-Anything-2 based video object tracking - [Installation Guide](../../examples/interactive_video_object_segmentation/README.md)
> 2. **Bounding Box Generation**: UPN-based bounding box generation - [Installation Guide](../../examples/detection/hbb/README.md)
> 3. **Interactive Detection & Segmentation**: Interactive object detection and segmentation with visual and text prompts - [Installation Guide](../../examples/detection/hbb/README.md)
> 4. **Smart Detection & Segmentation**: Object detection and segmentation with visual prompts, text prompts, and prompt-free modes - [Installation Guide](../../examples/grounding/yoloe/README.md)
> 5. **One-Click Training Platform**: Ultralytics framework-based training platform - [Installation Guide](../../examples/training/ultralytics/README.md)

### 1.1 Prerequisites

#### 1.1.1 Miniconda

**Step 0.** Download and install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 1.** Create a conda environment with Python 3.10 ~ 3.12 and activate it.

> [!NOTE]
> Other Python versions require compatibility verification on your own.

```bash
# CPU Environment [Windows/Linux/macOS]
conda create --name x-anylabeling-cpu python=3.10 -y
conda activate x-anylabeling-cpu

# CUDA 11.x Environment [Windows/Linux]
conda create --name x-anylabeling-cu11 python=3.11 -y
conda activate x-anylabeling-cu11

# CUDA 12.x Environment [Windows/Linux]
conda create --name x-anylabeling-cu12 python=3.12 -y
conda activate x-anylabeling-cu12
```

#### 1.1.2 Venv

In addition to Miniconda, you can also use Python's built-in `venv` module to create virtual environments. Here are the commands for creating and activating environments under different configurations:

```bash
# CPU [Windows/Linux/macOS]
python3.10 -m venv venv-cpu
source venv-cpu/bin/activate  # Linux/macOS
# venv-cpu\Scripts\activate    # Windows

# CUDA 12.x [Windows/Linux]
python3.12 -m venv venv-cu12
source venv-cu12/bin/activate  # Linux
# venv-cu12\Scripts\activate    # Windows

# CUDA 11.x [Windows/Linux]
python3.11 -m venv venv-cu11
source venv-cu11/bin/activate  # Linux
# venv-cu11\Scripts\activate    # Windows
```

> [!TIP]
> For faster dependency installation and a more modern Python package management experience, we strongly recommend using [uv](https://github.com/astral-sh/uv) as your package manager. uv provides significantly faster installation speeds and better dependency resolution capabilities.

### 1.2 Installation

#### 1.2.1 Pip Installation

You can easily install the latest stable version of X-AnyLabeling with the following commands:

```bash
# CPU [Windows/Linux/macOS]
pip install x-anylabeling-cvhub[cpu]

# CUDA 12.x is the default GPU option [Windows/Linux]
pip install x-anylabeling-cvhub[gpu]

# CUDA 11.x [Windows/Linux]
pip install x-anylabeling-cvhub[gpu-cu11]
```

After installation, you can verify it by running the following command:

```bash
xanylabeling checks   # Display system and version information
```

You can also run the following commands to get other information:

```bash
xanylabeling help     # Display help information
xanylabeling version  # Display version number
xanylabeling config   # Display configuration file path
```

After verification, you can run the application directly:

```bash
xanylabeling
```

### **2. Open export directory**

```
output_video/your_video/export/
```

### **3. Files auto-load**

Each `.jpg` pairs with `.json` for seamless annotation.

### **4. Modify manually if needed**

* Add missing boxes
* Delete false positives
* Export as COCO / YOLO / VOC if needed

---

# **11. Validation Checklist**

Before trusting your pipeline:

### ✔ GPU detected inside Docker

```bash
python -c "import paddle; paddle.device.is_compiled_with_cuda()"
```

### ✔ Region detection script loads

```bash
python region_detection.py --help
```

### ✔ Frames extracted

### ✔ JSON exported for each frame

### ✔ X-AnyLabeling opens directory successfully

---

# **12. Recommended Project Structure (Full)**

```
workspace/
├── data/                   # Input videos
├── output_video/           # Output of pipeline
├── scripts/
│   ├── region_detection.py
│   ├── utils.py
│   └── render.py
├── requirements.txt
├── Dockerfile (optional)
├── logs/
│   └── runtime.log
└── README.md
```

---
