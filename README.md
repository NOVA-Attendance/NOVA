# NOVA Face Recognition - Quick Start

Simple steps to enroll images, recognize from files, and run the webcam with live "Recognized in system" overlay.

## Prerequisites
- Python 3.9+ installed
- Internet access for first-time model downloads

## Project Layout (key paths)
- Data (unified folder recommended): `data/image_database/`
- Index output: `data/face_index.npz`, `data/labels.json`
- Scripts: `scripts/*.py`

## macOS Setup
1) Open Terminal and go to the project folder:
```bash
cd /Users/fareiscanoe/Desktop/NOVA
```
2) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Enroll Images (build the index)
- Put all images into the unified folder: `data/image_database/`
- Run enrollment:
```bash
python3 scripts/enroll_faces.py --images-dir data/image_database
```

### Recognize From Files
- Single file:
```bash
python3 scripts/recognize_image.py --image data/image_database/your_image.jpg --threshold 0.4
```
- Batch over a folder:
```bash
python3 scripts/recognize_image.py --batch data/image_database --threshold 0.4
```

### Webcam With Live Recognition
```bash
python3 scripts/facial_user_id_webcam.py --recognize-live --threshold 0.4
```
- If you have multiple cameras, add `--camera 1` (or 2).
- If the camera is blocked: System Settings → Privacy & Security → Camera → allow Terminal/IDE.

## Windows Setup
1) Open Command Prompt (or PowerShell) and go to the project folder:
```bat
cd C:\Users\<YourUser>\Desktop\NOVA
```
2) Create and activate a virtual environment:
```bat
python -m venv .venv
.venv\Scripts\activate
```
3) Install dependencies:
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Enroll Images (build the index)
- Put all images into: `data\image_database\`
- Run enrollment:
```bat
python scripts\enroll_faces.py --images-dir data\image_database
```

### Recognize From Files
- Single file:
```bat
python scripts\recognize_image.py --image data\image_database\your_image.jpg --threshold 0.4
```
- Batch over a folder:
```bat
python scripts\recognize_image.py --batch data\image_database --threshold 0.4
```

### Webcam With Live Recognition
```bat
python scripts\facial_user_id_webcam.py --recognize-live --threshold 0.4
```
- If multiple cameras, try `--camera 1`.
- If the window doesn’t show the camera, check Windows camera privacy settings.

## Tips
- Threshold tuning: 0.35–0.5 typical. Higher = stricter matches.
- Supported image types: jpg, jpeg, png, webp.
- Re-run enrollment after adding new images.

## Troubleshooting
- SSL/urllib3 warning on macOS is safe to ignore for local use.
- If you see NumPy/TensorFlow version conflicts, upgrade NumPy first:
```bash
pip install --upgrade numpy==1.26.4
pip install -r requirements.txt
```
- Camera permission issues: enable camera access for your Terminal/IDE in OS settings.
