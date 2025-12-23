import cv2
import numpy as np
from transformers.models.owlv2.image_processing_owlv2 import box_iou
from transformers.models.owlv2.modeling_owlv2 import box_area
from mtcnn import MTCNN


def detect_stream_box(frame, detector):
    """
    Returns (x1, y1, x2, y2) for the streamer camera box.
    Uses the largest detected face, then expands + refines with edges.
    """
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )
    if len(faces) == 0:
        return None

    # largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2.0, y + h / 2.0

    # 1) BIGGER initial webcam box (add margin)
    # tweak factors if needed; these already overshoot in your example
    box_h = int(h * 3.4)
    box_w = int(box_h * 4.0 / 3)  # assume ~4:3 webcam

    x1 = max(int(cx - box_w / 2), 0)
    y1 = max(int(cy - box_h / 2), 0)
    x2 = min(int(cx + box_w / 2), W - 1)
    y2 = min(int(cy + box_h / 2), H - 1)

    # 2) Snap the bottom edge to the real boundary (webcam → gameplay)
    #    by looking for the strongest horizontal edge in that region.
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # vertical derivative
    row_strength = np.mean(np.abs(gy[:, x1:x2]), axis=1)

    # search around the current bottom estimate
    search_from = max(0, y2 - int(0.5 * h))
    search_to   = min(H - 2, y2 + int(2.0 * h))
    if search_to > search_from:
        best_row = search_from + int(np.argmax(row_strength[search_from:search_to]))
        y2 = best_row

    # optional: if your overlays are always stuck to the top edge,
    # force y1 = 0 so you don’t miss any pixels at the top
    # y1 = 0

    return x1, y1, x2, y2

def detect_stream_box_mtcnn(frame, detector):
    """
    Same as detect_stream_box, but uses an MTCNN detector for face detection.
    Expects `detector` to be an instance of `mtcnn.MTCNN`.
    Returns (x1, y1, x2, y2), or None if no faces are found.
    """
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MTCNN returns list of dicts with 'box': [x, y, w, h]
    detections = detector.detect_faces(rgb) if detector is not None else []
    if len(detections) == 0:
        return None

    # choose the largest face by area
    def _area(box):
        bx, by, bw, bh = box
        return max(0, bw) * max(0, bh)

    x, y, w, h = max((d["box"] for d in detections), key=_area)

    # MTCNN boxes can be negative; clamp to image bounds
    x = max(0, x)
    y = max(0, y)
    w = max(0, w)
    h = max(0, h)

    if w == 0 or h == 0:
        return None

    cx, cy = x + w / 2.0, y + h / 2.0

    # 1) Bigger initial webcam box (add margin), matching original logic
    box_h = int(h * 3.4)
    box_w = int(box_h * 4.0 / 3)  # assume ~4:3 webcam

    x1 = max(int(cx - box_w / 2), 0)
    y1 = max(int(cy - box_h / 2), 0)
    x2 = min(int(cx + box_w / 2), W - 1)
    y2 = min(int(cy + box_h / 2), H - 1)

    # 2) Snap the bottom edge to the real boundary (webcam → gameplay)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # vertical derivative
    if x2 > x1:
        row_strength = np.mean(np.abs(gy[:, x1:x2]), axis=1)
    else:
        row_strength = np.mean(np.abs(gy), axis=1)

    # search around the current bottom estimate
    search_from = max(0, y2 - int(0.5 * h))
    search_to = min(H - 2, y2 + int(2.0 * h))
    if search_to > search_from:
        best_row = search_from + int(np.argmax(row_strength[search_from:search_to]))
        y2 = best_row

    return x1, y1, x2, y2


# ---- step 1: get candidate boxes from Canny edges ---------------------------

def get_candidate_regions_from_edges(frame, min_area_ratio=0.05):
    """
    Use grayscale + Canny to find large rectangular-ish regions.
    Returns a list of (x, y, w, h) bounding boxes.
    """
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # reduce noise so Canny gives cleaner edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #blur = gray
    edges = cv2.Canny(blur, 50, 150)

    # thicken edges so contours close
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    frame_area = W * H
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area_ratio * frame_area:
            continue  # ignore tiny junk
        regions.append((x, y, w, h))

    # fallback: if nothing big was found, treat entire frame as one region
    if not regions:
        regions = [(0, 0, W, H)]

    return regions

# ---- step 2: pick region with highest face-coverage -------------------------

def choose_streamer_region_by_faces(frame, regions, face_detector):
    """
    For each candidate region, run face detection and compute:
      face_area_sum / region_area
    Returns the region with the highest ratio, or None if no faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_region = None
    best_ratio = 0.0

    for (x, y, w, h) in regions:
        roi_gray = gray[y:y+h, x:x+w]

        faces = face_detector.detectMultiScale(
            roi_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            continue

        face_area = sum((fw * fh) for (_, _, fw, fh) in faces)
        region_area = float(w * h)
        ratio = face_area / region_area

        if ratio > best_ratio:
            best_ratio = ratio
            best_region = (x, y, w, h)

    return best_region
    

# --- split frame -------------------------------------------------------------

def split_frame(frame, detector, mask_mode="black"):
    #num_region = 0
    if detector is not None:
        #print("Using face detector")
        #regions = get_candidate_regions_from_edges(frame)
        #num_region = len(regions)
        #box = choose_streamer_region_by_faces(frame, regions, detector)
        box = detect_stream_box_mtcnn(frame, detector)
    
    else:
        #print("Using edges detector")
        box = None
    

    if box is None:
        return None, frame.copy()

    print("box:", box)
    x1, y1, x2, y2 = box

    # crop streamer camera
    streamer = frame[y1:y2, x1:x2].copy()

    # gameplay with streamer area blacked/blurred
    gameplay = frame.copy()
    if mask_mode == "black":
        gameplay[y1:y2, x1:x2] = 0
    elif mask_mode == "blur":
        gameplay[y1:y2, x1:x2] = cv2.GaussianBlur(gameplay[y1:y2, x1:x2], (15, 15), 0)
    else:
        raise ValueError(f"Invalid mask mode: {mask_mode}")

    return streamer, gameplay



def process_video(
    input_path,
    gameplay_path,
    streamer_path,
    stream_box=(768, 512),      # streamer box size typically matched with svd output frame size
    mask_mode="black",     # "black" or "blur"
    face_detector:bool=True,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input: {input_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gameplay_writer = cv2.VideoWriter(gameplay_path, fourcc, fps, (W, H))
    sw, sh = stream_box
    streamer_writer = cv2.VideoWriter(streamer_path, fourcc, fps, (sw, sh))

    detector = None
    print("face detector args:", face_detector)
    if face_detector:
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        detector = MTCNN()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        streamer, gameplay = split_frame(frame, detector, mask_mode)
        if streamer is None or streamer.size == 0:
            streamer = np.zeros((sh, sw, 3), dtype=np.uint8)
        else:
            streamer = cv2.resize(streamer, (sw, sh))

        streamer_writer.write(streamer)
        gameplay_writer.write(gameplay)

    cap.release()
    gameplay_writer.release()
    streamer_writer.release()