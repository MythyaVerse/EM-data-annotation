# CVAT Local Setup Guide for Text Detection Pipeline

## Overview
This guide sets up CVAT (Computer Vision Annotation Tool) locally to integrate with your text detection pipeline for human-in-the-loop annotation correction.

## Architecture
```
Text Detection Pipeline → CVAT Import → Human Correction → Export Corrected Data
```

## Prerequisites
- Docker Desktop for Windows installed and running
- Python 3.8+
- At least 8GB RAM
- 20GB free disk space

## Step 1: Install CVAT Using Docker

### 1.1 Create CVAT Directory
```bash
mkdir cvat_local
cd cvat_local
```

### 1.2 Download CVAT Docker Compose
```bash
# Clone CVAT repository
git clone https://github.com/cvat-ai/cvat
cd cvat
```

### 1.3 Start CVAT Services
```bash
# For Windows (PowerShell)
docker-compose up -d
```

This will start:
- CVAT web interface (http://localhost:8080)
- PostgreSQL database
- Redis cache
- CVAT backend API

### 1.4 Create Superuser Account
```bash
# Create admin user
docker exec -it cvat_server bash -ic 'python3 manage.py createsuperuser'
```

Enter credentials:
- Username: admin
- Email: admin@localhost
- Password: your_secure_password

### 1.5 Verify Installation
Open browser and navigate to: http://localhost:8080
Login with your admin credentials.

## Step 2: Install CVAT Python SDK

```bash
# In your project directory (d:\Projects\Data Annotation)
pip install cvat-sdk
pip install cvat-cli
```

## Step 3: Configuration

### 3.1 Create CVAT Configuration File
Create `cvat_config.json` in your project root:

```json
{
  "cvat_host": "http://localhost:8080",
  "cvat_username": "admin",
  "cvat_password": "your_secure_password",
  "organization": "",
  "project_name": "Text Detection Correction",
  "task_prefix": "text_detection_"
}
```

## Step 4: Project Structure

Your enhanced project structure:
```
Data Annotation/
├── run_text_detection.py          # Your detection pipeline
├── text_detection.py              # Detection model
├── cvat_integration/
│   ├── __init__.py
│   ├── format_converter.py        # Convert detection JSON → CVAT format
│   ├── cvat_uploader.py          # Upload to CVAT
│   ├── cvat_exporter.py          # Export corrected annotations
│   └── workflow.py               # End-to-end orchestration
├── cvat_config.json              # CVAT connection settings
├── CVAT_SETUP.md                 # This file
└── output/                       # Detection outputs
    └── text_detection_updated/
        └── [video_name]/
            ├── annotated_video_detection.mp4
            ├── frames_and_json/
            │   ├── frame_00000.jpg
            │   ├── frame_00000.json
            │   └── ...
            └── detection_statistics.json
```

## Step 5: Workflow

### 5.1 Run Text Detection
```bash
python run_text_detection.py
```
This generates frames + JSON annotations in `output/text_detection_updated/[video_name]/frames_and_json/`

### 5.2 Convert & Upload to CVAT
```bash
python -m cvat_integration.workflow --video-name [video_name] --action upload
```
This will:
1. Convert detection JSON to CVAT format
2. Create a new task in CVAT
3. Upload frames
4. Import pre-annotations (your detections)

### 5.3 Human Correction in CVAT
1. Open http://localhost:8080
2. Navigate to the created task
3. Annotators correct/improve bounding boxes
4. Save changes

### 5.4 Export Corrected Annotations
```bash
python -m cvat_integration.workflow --video-name [video_name] --action export
```
This exports corrected annotations in your original JSON format.

## Step 6: CVAT API Endpoints

Key endpoints used by integration scripts:
- `POST /api/projects` - Create project
- `POST /api/tasks` - Create annotation task
- `POST /api/tasks/{id}/data` - Upload images
- `POST /api/tasks/{id}/annotations` - Import pre-annotations
- `GET /api/tasks/{id}/annotations` - Export annotations

## Troubleshooting

### Docker Issues
```bash
# Check if CVAT containers are running
docker ps

# View logs
docker-compose logs -f cvat_server

# Restart services
docker-compose restart
```

### Connection Issues
- Ensure Docker Desktop is running
- Check http://localhost:8080 is accessible
- Verify credentials in `cvat_config.json`

### Port Conflicts
If port 8080 is taken, edit `docker-compose.yml`:
```yaml
services:
  cvat_proxy:
    ports:
      - "8090:80"  # Change to 8090 or any free port
```

## Security Notes

- Default setup is for LOCAL USE ONLY
- Do not expose CVAT to the internet without proper security
- Change default passwords
- Use environment variables for production credentials

## Advanced Configuration

### Enable GPU Support (Optional)
Edit `docker-compose.yml`:
```yaml
services:
  cvat_server:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### Persistent Storage
Data is stored in Docker volumes:
- `cvat_db` - PostgreSQL database
- `cvat_data` - Uploaded images and annotations

## Next Steps

After setup:
1. Test the workflow with a sample video
2. Configure user accounts for annotators
3. Set up quality control workflows
4. Integrate with your training pipeline

## References

- CVAT Documentation: https://opencv.github.io/cvat/
- CVAT SDK: https://github.com/cvat-ai/cvat/tree/develop/cvat-sdk
- API Reference: https://opencv.github.io/cvat/docs/api_sdk/
