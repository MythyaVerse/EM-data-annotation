# Setup Guide: PaddleOCR Video Annotation Pipeline

## Hardware Requirements

**Minimum Specs:**
- **GPU:** NVIDIA GPU with 6GB+ VRAM (GTX 1660 Ti or better)
- **RAM:** 16GB system memory
- **Storage:** 10GB free space
- **CUDA:** Compatible NVIDIA GPU with CUDA support

**Recommended Specs:**
- **GPU:** NVIDIA RTX 3060 or better (8GB+ VRAM)
- **RAM:** 32GB system memory
- **Storage:** SSD with 20GB+ free space

---

## Quick Setup

### 1. Pull PaddleOCR Docker Image

```bash
docker pull paddlepaddle/paddle:latest-gpu-cuda11.7-cudnn8
```

Or for CUDA 12.x:
```bash
docker pull paddlepaddle/paddle:latest-gpu-cuda12.0-cudnn8
```

### 2. Install NVIDIA Container Toolkit (if not installed)

**Windows (WSL2):**
```bash
wsl --install
```

**Linux:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Run Docker Container

```bash
docker run --gpus all -it --rm \
  -v "d:\Projects\Data Annotation Tool:/workspace" \
  paddlepaddle/paddle:latest-gpu-cuda11.7-cudnn8 /bin/bash
```

### 4. Install Dependencies (inside container)

```bash
pip install paddleocr paddlepaddle-gpu opencv-python-headless numpy
```

### 5. Setup Project Structure

```bash
cd /workspace
mkdir -p data output_video
```

### 6. Run the Pipeline

Place your video in the `data/` folder, then:

```bash
python region_detection.py --video your_video.mp4 --fps 1
```

**Options:**
- `--video` or `-v`: Video filename (must be in `data/` directory)
- `--fps` or `-f`: Frames per second to extract (default: 1)

---

## Output Structure

```
output_video/
└── triangles/                      # Video name (without extension)
    ├── annotated_video.mp4        # Video with visual annotations
    ├── video_annotations.json     # Complete annotations in JSON format
    ├── classes.txt                # List of all detectable classes
    └── export/                    # X-AnyLabeling compatible directory
        ├── frame_0001.jpg         # Original extracted frame
        ├── frame_0001.json        # X-AnyLabeling annotation file
        ├── frame_0002.jpg
        ├── frame_0002.json
        └── ...
```

---

## Using X-AnyLabeling

1. Download X-AnyLabeling: https://github.com/CVHub520/X-AnyLabeling
2. Open X-AnyLabeling
3. File > Open Dir
4. Navigate to `output_video/your_video/export/`
5. Annotations load automatically for manual correction

---

## Troubleshooting

**GPU not detected:**
```bash
nvidia-smi  # Check GPU availability
docker run --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi
```

**Out of memory:**
- Reduce video resolution
- Process fewer frames (`--fps 0.5` for 1 frame every 2 seconds)

**Slow processing:**
- Expect ~2-5 seconds per frame on RTX 3060
- Use SSD for faster I/O
