"""Smart City Vision: YOLOv8 + ByteTrack + counting + pedestrian zone + optional ANPR.

Portfolio demonstration only. People and plate observations can be erroneous; don't use for
surveillance decisions, enforcement, public safety alerts, or identity verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

VEHICLES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
PERSON = 0
TRACK_CLASSES = [0, 2, 3, 5, 7]
COLOR_BG = (32, 19, 12)
COLOR_TEAL = (226, 208, 37)
COLOR_BLUE = (243, 142, 51)
COLOR_YELLOW = (90, 211, 255)
COLOR_GREEN = (118, 220, 83)
COLOR_RED = (96, 99, 255)
COLOR_WHITE = (246, 247, 250)


def mask_plate(text: str) -> str:
    """Hide identifying characters on public demo outputs."""
    letters = ''.join(c for c in text.upper() if c.isalnum())
    if not letters:
        return 'PLATE DETECTED'
    return letters[:2] + '*' * max(3, len(letters) - 2)


def crosses_line(previous: tuple[int, int], current: tuple[int, int],
                 orientation: str, position: int, margin: int = 3) -> Optional[str]:
    """Return crossing direction if point moves from outside margin on one side to other.

    Crossing within margin is handled by maintaining the last *stable* side separately.
    """
    a = previous[0] if orientation == 'vertical' else previous[1]
    b = current[0] if orientation == 'vertical' else current[1]
    if a < position - margin and b > position + margin:
        return 'right' if orientation == 'vertical' else 'down'
    if a > position + margin and b < position - margin:
        return 'left' if orientation == 'vertical' else 'up'
    return None


def stable_side(point: tuple[int, int], orientation: str, position: int, margin: int = 5) -> int:
    value = point[0] if orientation == 'vertical' else point[1]
    return -1 if value < position - margin else (1 if value > position + margin else 0)


def in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    a, b, c, d = rect
    return a <= x <= c and b <= y <= d


def put_text(img, message: str, position: tuple[int, int], scale: float = .62,
             color=COLOR_WHITE, thickness: int = 2):
    cv2.putText(img, message, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, message, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def pill(img, msg: str, x: int, y: int, color=COLOR_TEAL, scale: float = .55):
    tw, th = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    left, top, right, bottom = max(x, 0), max(y - th - 9, 0), min(img.shape[1] - 1, x + tw + 16), min(img.shape[0] - 1, y + 9)
    if right <= left or bottom <= top:
        return
    cv2.rectangle(img, (left, top), (right, bottom), (24, 24, 24), -1)
    cv2.rectangle(img, (left, top), (right, bottom), color, 1)
    put_text(img, msg, (left + 7, min(img.shape[0] - 4, y)), scale, color, 1)


def alpha_rectangle(img, x1, y1, x2, y2, color=COLOR_BG, opacity=.80):
    x1, x2 = max(0, x1), min(img.shape[1], x2)
    y1, y2 = max(0, y1), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = cv2.addWeighted(roi, 1 - opacity, np.full_like(roi, color), opacity, 0)


def draw_box(img, bounds, label: str, color=COLOR_TEAL, font_scale=.51):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bounds]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    tag_top = max(0, y1 - th - 15)
    cv2.rectangle(img, (x1, tag_top), (min(w - 1, x1 + tw + 12), tag_top + th + 12), color, -1)
    cv2.putText(img, label, (x1 + 5, tag_top + th + 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BG, 1, cv2.LINE_AA)


def make_zone(args, width: int, height: int):
    if args.no_zone:
        return None
    a, b, c, d = args.zone
    return (int(a * width), int(b * height), int(c * width), int(d * height))


def extract_ocr_text(reader, crop) -> str:
    if reader is None or crop.size == 0:
        return ''
    try:
        candidates = reader.readtext(crop, detail=1, paragraph=False)
        good = [str(text) for _, text, confidence in candidates if float(confidence) >= .30]
        return ''.join(char for char in ''.join(good).upper() if char.isalnum())[:15]
    except Exception as exc:
        print(f'[OCR warning] {exc}')
        return ''


def draw_dashboard(frame, frame_index: int, fps: float, vehicle_counts: Counter,
                   visible_vehicles: int, visible_people: int, unique_people: int,
                   plates_found: int, zone_activity: int, video_name: str):
    h, w = frame.shape[:2]
    sm = min(1.0, w / 1024)
    head_h = max(92, int(114 * sm))
    alpha_rectangle(frame, 0, 0, w, head_h, opacity=.89)
    cv2.rectangle(frame, (0, head_h - 3), (w, head_h), COLOR_TEAL, -1)
    put_text(frame, 'CITYVISION  /  SMART TRAFFIC ANALYTICS', (14, max(24, int(33 * sm))), .68 * sm, COLOR_TEAL, 2)
    put_text(frame, f'SOURCE  {video_name[:30]}    |    TIME  {frame_index / fps:05.1f}s',
             (14, max(43, int(53 * sm))), .40 * sm, (205, 208, 217), 1)
    values = [
        ('CROSSED', str(sum(vehicle_counts.values()))),
        ('VISIBLE VEHICLES', str(visible_vehicles)),
        ('PEOPLE', str(visible_people)),
        ('PLATES', str(plates_found)),
    ]
    start_y = int(77 * sm)
    for i, (label, value) in enumerate(values):
        x = 14 + i * int((w - 28) / 4)
        put_text(frame, f'{label}: {value}', (x, start_y + 10), .46 * sm, COLOR_WHITE, 1)
    foot_h = max(52, int(63 * sm))
    alpha_rectangle(frame, 0, h - foot_h, w, h, opacity=.84)
    cls_line = '   '.join(f'{label.upper()} {vehicle_counts[label]}' for label in ['car','motorcycle','bus','truck'])
    put_text(frame, cls_line, (14, h - max(25, int(37 * sm))), .43 * sm, COLOR_WHITE, 1)
    status = f'DEMO ZONE: {zone_activity} PEDESTRIAN(S)' if zone_activity else 'TRACKING ACTIVE / PRIVACY-READY DEMO'
    put_text(frame, status, (14, h - max(8, int(14 * sm))), .40 * sm,
             COLOR_YELLOW if zone_activity else COLOR_TEAL, 1)


@dataclass
class PlateStatus:
    text: str = ''
    frame: int = 0


def build_parser():
    p = argparse.ArgumentParser(description='Smart City YOLOv8 traffic demo / optional license-plate detector')
    p.add_argument('--input', default='videos', help='Input video file or folder of videos')
    p.add_argument('--output', default='outputs', help='Output folder for annotated mp4 and reports')
    p.add_argument('--model', default='yolov8n.pt', help='COCO detector (downloads via Ultralytics if absent)')
    p.add_argument('--conf', type=float, default=.32)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--tracker', default='bytetrack.yaml')
    p.add_argument('--device', default=None, help='Examples: cpu, 0 for first CUDA GPU')
    p.add_argument('--max-width', type=int, default=1280, help='Resize wide videos for speed; 0 keeps original')
    p.add_argument('--line-orientation', choices=('vertical', 'horizontal'), default='vertical')
    p.add_argument('--line-pos', type=float, default=.50, help='Normalized count-line position, 0..1')
    p.add_argument('--zone', type=float, nargs=4, default=(.30, .40, .70, .88),
                   metavar=('LEFT','TOP','RIGHT','BOTTOM'), help='Normalized demo pedestrian zone')
    p.add_argument('--no-zone', action='store_true')
    p.add_argument('--plates', action='store_true', help='Enable plate detector, separate weights required')
    p.add_argument('--plate-model', default='models/license_plate_detector.pt')
    p.add_argument('--plate-every', type=int, default=12, help='Process candidate vehicles every N frames')
    p.add_argument('--ocr', action='store_true', help='Enable optional EasyOCR (runs on CPU by default)')
    p.add_argument('--show', action='store_true', help='Open OpenCV preview window; Q to stop')
    p.add_argument('--max-frames', type=int, default=0, help='Optional max processed frames per video')
    return p


def validate_args(args):
    if not 0 < args.line_pos < 1:
        raise ValueError('--line-pos must be between 0 and 1')
    if any(not 0 <= x <= 1 for x in args.zone) or args.zone[0] >= args.zone[2] or args.zone[1] >= args.zone[3]:
        raise ValueError('--zone must be four ordered fractions [x1 y1 x2 y2] from 0..1')
    if args.plate_every < 1:
        raise ValueError('--plate-every must be >= 1')
    if args.ocr and not args.plates:
        raise ValueError('--ocr needs --plates')
    if args.plates and not Path(args.plate_model).is_file():
        raise FileNotFoundError(f'Missing plate weights: {args.plate_model}. Run python download_plate_model.py first.')
    if args.max_width < 0:
        raise ValueError('--max-width must be >= 0')


def process_video(video_path: Path, args):
    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'[SKIP] Cannot open: {video_path}')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 10 or height < 10:
        cap.release()
        print(f'[SKIP] Invalid dimensions: {video_path}')
        return
    if args.max_width and width > args.max_width:
        height = round(height * args.max_width / width)
        width = args.max_width
    width -= width % 2
    height -= height % 2
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'{video_path.stem}_smartcity.mp4'
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Could not open output writer: {dst}')

    # A new YOLO instance per video prevents tracker IDs from crossing video boundaries.
    model = YOLO(args.model)
    plate_model = YOLO(args.plate_model) if args.plates else None
    reader = None
    if args.ocr:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError('Install OCR extras: pip install easyocr') from exc
        reader = easyocr.Reader(['en'], gpu=False)

    line_px = int((width if args.line_orientation == 'vertical' else height) * args.line_pos)
    rect = make_zone(args, width, height)
    prev_side = {}
    counted_ids = set()
    trails = defaultdict(lambda: deque(maxlen=22))
    people_seen = set()
    counted = Counter()
    plate_cache: dict[int, PlateStatus] = {}
    plate_seen = set()
    events = []
    frames_processed = 0
    print(f'[START] {video_path.name}: {width}x{height} @ {fps:.1f}fps  plates={args.plates} ocr={args.ocr}')
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames_processed += 1
            clean_frame = frame.copy()  # plate recognition must run on pixels without overlays
            result = model.track(clean_frame, persist=True, tracker=args.tracker,
                                 classes=TRACK_CLASSES, imgsz=args.imgsz, conf=args.conf,
                                 device=args.device, verbose=False)[0]
            visible_v, visible_p, zone_people = 0, 0, 0
            tracked_vehicles = []
            if result.boxes is not None and result.boxes.id is not None:
                coords = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                for bb, tid, cat, score in zip(coords, ids, classes, confs):
                    x1, y1, x2, y2 = map(int, bb)
                    cx, cy = (x1 + x2)//2, (y1+y2)//2
                    point = (cx, y2) # bottom-middle point is appropriate for line + demo zone
                    color = COLOR_TEAL if cat in VEHICLES else COLOR_YELLOW
                    kind = VEHICLES.get(cat, 'person')
                    if cat == PERSON:
                        visible_p += 1
                        people_seen.add(tid)
                        if rect is not None and in_rect(point, rect):
                            zone_people += 1
                    else:
                        visible_v += 1
                        tracked_vehicles.append((tid, (x1, y1, x2, y2), score))
                        side = stable_side(point, args.line_orientation, line_px)
                        last_side = prev_side.get(tid)
                        if side != 0:
                            if last_side is not None and last_side != side and tid not in counted_ids:
                                direction = ('left' if side < 0 else 'right') if args.line_orientation == 'vertical' else ('up' if side < 0 else 'down')
                                counted[kind] += 1
                                counted_ids.add(tid)
                                events.append({'frame': frames_processed, 'seconds': round(frames_processed/fps, 2),
                                               'event': 'vehicle_crossed', 'track_id': tid, 'class': kind,
                                               'direction': direction, 'masked_plate': plate_cache.get(tid, PlateStatus()).text})
                            prev_side[tid] = side
                    trails[tid].append((cx, cy))
                    draw_box(frame, (x1, y1, x2, y2), f'{kind.upper()} #{tid} {score:.2f}', color)
                    for previous, current in zip(list(trails[tid])[:-1], list(trails[tid])[1:]):
                        cv2.line(frame, previous, current, color, 2, cv2.LINE_AA)

            # Draw an operator-defined DEMO watch zone, not a legal crosswalk classification.
            if rect is not None:
                x1, y1, x2, y2 = rect
                zone_color = COLOR_YELLOW if zone_people else COLOR_BLUE
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_color, -1)
                frame = cv2.addWeighted(frame, .94, overlay, .06, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 2)
                pill(frame, 'DEMO PEDESTRIAN WATCH ZONE', x1 + 4, y1 + 23, zone_color, .48)

            if args.line_orientation == 'vertical':
                cv2.line(frame, (line_px, int(height*.19)), (line_px, height - 75), COLOR_GREEN, 2)
                pill(frame, 'COUNT LINE', line_px + 5, int(height*.19) + 20, COLOR_GREEN, .45)
            else:
                cv2.line(frame, (10, line_px), (width - 10, line_px), COLOR_GREEN, 2)
                pill(frame, 'COUNT LINE', 16, line_px - 6, COLOR_GREEN, .45)

            if plate_model is not None and frames_processed % args.plate_every == 0:
                # Limit to the largest 4 vehicles per sampled frame to avoid slow CPU runs.
                top = sorted(tracked_vehicles, key=lambda it: (it[1][2] - it[1][0]) * (it[1][3] - it[1][1]), reverse=True)[:4]
                for tid, (x1, y1, x2, y2), _ in top:
                    if (x2-x1) < 75 or (y2-y1) < 45:
                        continue
                    roi = clean_frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                    if roi.size == 0:
                        continue
                    try:
                        plates = plate_model.predict(roi, imgsz=640, conf=.33, verbose=False)[0]
                    except Exception as exc:
                        print(f'[PLATE model warning] {exc}')
                        continue
                    for pb in plates.boxes.xyxy.cpu().numpy() if plates.boxes is not None else []:
                        a, b, c, d = map(int, pb)
                        a, b = max(a, 0), max(b, 0)
                        c, d = min(c, roi.shape[1]), min(d, roi.shape[0])
                        if c - a < 12 or d - b < 8:
                            continue
                        plate_crop = roi[b:d, a:c].copy()
                        # OCR is opt-in. Keep only MASKED text, never raw OCR in logs or overlay.
                        text = mask_plate(extract_ocr_text(reader, plate_crop)) if args.ocr else 'PLATE DETECTED'
                        plate_cache[tid] = PlateStatus(text, frames_processed)
                        plate_seen.add(tid)
                        px1, py1, px2, py2 = x1 + a, y1 + b, x1 + c, y1 + d
                        # Mosaic detected crop on this sampled frame; other frames MUST be reviewed before public sharing.
                        patch = frame[py1:py2, px1:px2]
                        if patch.size:
                            tiny = cv2.resize(patch, (max(1, patch.shape[1] // 12), max(1, patch.shape[0] // 12)), interpolation=cv2.INTER_AREA)
                            patch[:] = cv2.resize(tiny, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)
                        cv2.rectangle(frame, (px1,py1), (px2,py2), COLOR_YELLOW, 2)
                        pill(frame, f'{text}', max(0, px1-8), max(16,py1-4), COLOR_YELLOW, .41)
                        break
            draw_dashboard(frame, frames_processed, fps, counted, visible_v, visible_p,
                           len(people_seen), len(plate_seen), zone_people, video_path.name)
            writer.write(frame)
            if args.show:
                try:
                    cv2.imshow('CityVision Smart Traffic', frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                except cv2.error:
                    print('[INFO] Preview GUI unavailable; continuing headless.')
                    args.show = False
            if frames_processed % 60 == 0:
                print(f'  {frames_processed} frames | {sum(counted.values())} crossings | {len(people_seen)} unique person tracks')
            if args.max_frames and frames_processed >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    csv_path = out_dir / f'{video_path.stem}_events.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        fields = ['frame','seconds','event','track_id','class','direction','masked_plate']
        dw = csv.DictWriter(f, fieldnames=fields)
        dw.writeheader()
        dw.writerows(events)
    summary = {
        'source': video_path.name,
        'annotated_video': dst.name,
        'frames_processed': frames_processed,
        'video_seconds_processed': round(frames_processed / fps, 2),
        'vehicle_crossings_total': sum(counted.values()),
        'vehicle_crossings_by_type': dict(counted),
        'unique_person_track_ids_observed': len(people_seen),
        'vehicle_track_ids_crossed': len(counted_ids),
        'plate_track_ids_detected': len(plate_seen),
        'plate_ocr_enabled': bool(args.ocr),
        'privacy': 'Detected plate crops are pixelated on sampled frames; OCR masked; inspect/redact whole video before publishing',
        'limitations': 'Counts are track-based estimates; people track counts may double-count after ID swaps; not calibrated for traffic violations or identity decisions.',
    }
    js_path = out_dir / f'{video_path.stem}_summary.json'
    js_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[DONE] Video: {dst}\n       Events: {csv_path}\n       Summary: {js_path}')
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    source = Path(args.input)
    if source.is_file():
        videos = [source]
    elif source.is_dir():
        videos = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'})
    else:
        raise FileNotFoundError(f'Input not found: {source}')
    if not videos:
        raise SystemExit(f'No video files found in {source}. Download a traffic clip and place it in videos/.')
    for vid in videos:
        process_video(vid, args)


if __name__ == '__main__':
    main()
