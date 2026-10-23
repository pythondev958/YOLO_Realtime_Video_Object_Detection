**CityVision | Smart City Traffic Analytics (YOLOv8)
**

A portfolio-friendly Python computer-vision demo based on an existing YOLOv8 video detector.
![Stack: Python, Ultralytics YOLOv8, OpenCV](https://img.shields.io/badge/Python-YOLOv8%20%2B%20OpenCV-blue)
Features
Detect cars, motorcycles, buses, trucks and pedestrians with pretrained YOLOv8 COCO weights.
Track objects frame-to-frame with ByteTrack and short visual trails.
Count vehicles crossing a configurable line and show per-class totals.
Draw an operator-defined pedestrian watch zone (demo UI, not crosswalk inference).
Animated-looking high-contrast dashboard, annotated MP4, `events.csv`, and `summary.json`.
Optional license-plate detection via separate, third-party YOLOv8 plate model.
Optional EasyOCR text extraction. Plate strings are masked and detected plate regions are pixelated on sampled frames before saving the video (review every frame before publication).
Operates on a folder of MP4 videos; no GPU needed for the base demo, though CPU processing can be slow.
> Scope: This is a demonstration, NOT a validated traffic-safety, speed-enforcement, or identity-recognition system. COCO YOLOv8 does NOT detect license plates on its own. Counting accuracy depends on camera angle, occlusions and tracker ID stability. Stock footage may be too wide-angle for legible plates. The plate model has not been benchmarked or verified for your footage.
1. Windows PowerShell install
```powershell
cd D:\Personal\github_projects\YOLO_Realtime_Video_Object_Detection
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir videos, outputs, models -ErrorAction SilentlyContinue
```
Use Python 3.11 for broad compatibility; `py -3.11 --version` can verify it is installed. Python 3.12 may also work with compatible dependencies.
2. Get sample traffic footage
Recommended: Pexels — Busy Urban Intersection with Vehicles and Pedestrians. 1920×1080, 12 s, free download; supports vehicles and pedestrians.
Pexels — Urban Traffic at Intersection with Stoplights, 1920×1080, 9 s.
Pixabay — City, Urban, Traffic, 1920×1080.
Use Download on their site and check the creator/site license for intended use. Rename to `traffic.mp4` and put it inside `videos/`. For number-plate OCR, use a consented, close-range, stationary camera clip; wide intersection stock footage rarely gives enough pixels to read plates accurately.
3. Run
Start with the default demo (cars + people + ByteTrack + counting):
```powershell
python smart_city.py --input videos/traffic.mp4 --line-orientation vertical --line-pos 0.50
```
For traffic moving top to bottom on the video, use a horizontal line:
```powershell
python smart_city.py --input videos/traffic.mp4 --line-orientation horizontal --line-pos 0.63 --show
```
`--show` opens a preview window; omit it for batch/headless processing.
4. Optional plate detector (separate pretrained model)
```powershell
python download_plate_model.py
python smart_city.py --input videos/traffic.mp4 --plates --plate-every 12
```
A SHA256-pinned (~6.25 MB) pretrained third-party checkpoint is downloaded from Koushim/yolov8-license-plate-detection on Hugging Face, only when you explicitly run the download step. Review third-party model trust and license before use; pretrained `.pt` files deserialize Python/PyTorch objects.
Enable OCR only if needed (larger installation, slower CPU):
```powershell
pip install -r requirements-ocr.txt
python smart_city.py --input videos/traffic.mp4 --plates --ocr --plate-every 15
```
Privacy defaults: Plate crops detected by the optional model are pixelated on sampled frames only (every `--plate-every` frames); this does not guarantee redaction across the entire video. OCR characters are masked (first 2 visible, rest asterisks) and raw text is not logged. Face blurring is not included. Review and blur any visible identifiers in every frame before publication, use appropriately licensed/consented footage, and avoid identifiable bystanders without permission. For public demos, consider running without the plate feature or use synthetic plates.
5. Output files
```text
outputs/
  traffic_smartcity.mp4        # annotated HUD video
  traffic_events.csv          # counted vehicles: time, tracking ID, class, direction
  traffic_summary.json        # totals + limitations
```
Counts represent objects that crossed the virtual line, not total detections or legally certified traffic counts. `unique_person_track_ids_observed` is an estimated number of tracked IDs, not proof of unique individuals.
Advanced CLI examples
```powershell
# Run on all files in videos/ (one tracker instance per video)
python smart_city.py --input videos

# Faster testing on CPU: first 120 frames, 960px output width
python smart_city.py --input videos/traffic.mp4 --max-frames 120 --max-width 960 --device cpu

# Adjust demo watch-zone rectangle using fractions [left top right bottom]
python smart_city.py --input videos/traffic.mp4 --zone 0.25 0.40 0.60 0.90

# Turn off watch zone if the clip has no pedestrians
python smart_city.py --input videos/traffic.mp4 --no-zone
```
Development notes
YOLOv8 pretrained object labels: person=0, car=2, motorcycle=3, bus=5, truck=7 (COCO dataset).
Input clips should be recorded from a fixed camera for meaningful crossing counts.
CPU speed is highly dependent on video dimensions. YOLOv8 nano is the default to keep it practical on most laptops.
Optional plate OCR is not a guaranteed ANPR system. Plate recognizability depends heavily on distance, focus, angle and lighting.
Model licensing: Ultralytics is distributed under AGPL-3.0 or a separate enterprise license. Review its requirements before integrating into proprietary commercial software. Stock-video licensing is independent.
Do not commit downloaded model weights, licensed videos, confidential footage, or generated data into a public GitHub repository unless rights permit.
Clean GitHub update
```powershell
git status
git add smart_city.py download_plate_model.py README.md requirements.txt requirements-ocr.txt .gitignore
# Git commit / push only after testing the output and reviewing all included files
git commit -m "Add smart city traffic tracking, counting, and optional plate detection"
git push origin main
```
Further enhancements
Camera calibration for approximate speeds, longer multi-camera tracking, synthetic plate dataset tests, incident review dashboard, interactive React frontend with FastAPI, aggregated traffic charts and proper privacy-preserving face redaction.
