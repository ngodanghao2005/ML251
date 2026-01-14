import os
import cv2
import time
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR = "images"
LABEL_DIR = "labels"
OUTPUT_DIR = "results_predict"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

MIN_NEIGHBORS = 7
SCALE_FACTOR = 1.1
MIN_SIZE = (30, 30)
IOU_THRESHOLD = 0.5

# ----------------------------------------------------------------------
# IoU computation
# ----------------------------------------------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0

# ----------------------------------------------------------------------
# YOLO label → pixel bounding box
# ----------------------------------------------------------------------
def yolo_to_bbox(line, img_w, img_h):
    _, xc, yc, w, h = map(float, line.split())

    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h

    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)

    return (x1, y1, x2, y2)

def load_ground_truth(label_path, img_w, img_h):
    """
    Load all ground-truth bounding boxes for one image
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            gt_boxes.append(yolo_to_bbox(line, img_w, img_h))
    return gt_boxes

# ------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------
def load_images(data_dir):
    image_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jpg")
    ]
    return image_files

# ------------------------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# ------------------------------------------------------------------------------
# Haar Cascade Detection Pipeline (Parameter Tuning) + Evaluation Pipeline
# ------------------------------------------------------------------------------
def haar_detection_pipeline(image_files):
    print("Classic ML Pipeline: Haar Cascade")
    print("=" * 50)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TP, FP, FN = 0, 0, 0
    start_time = time.time()
    total_faces_detected = 0
        
    # Output folder for this experiment
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_path in tqdm(image_files):
        img, gray = preprocess_image(img_path)
        if img is None:
            continue

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )
            
        pred_boxes = [
            (x, y, x + bw, y + bh)
            for (x, y, bw, bh) in faces
        ]

        total_faces_detected += len(faces)
            
        # ---------------- Visualization ----------------
        for (x, y, w, h) in faces:
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        # ---------------- Save Result ----------------
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, img)
            
        # Load ground truth
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, img_name + ".txt")
        gt_boxes = load_ground_truth(label_path, img.shape[1], img.shape[0])

        matched_gt = set()
            
        for pred_box in pred_boxes:
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                if compute_iou(pred_box, gt_box) >= IOU_THRESHOLD:
                    TP += 1
                    matched_gt.add(i)
                    matched = True
                    break

            if not matched:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    end_time = time.time()
        
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Total faces detected: {total_faces_detected}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    image_files = load_images(DATA_DIR)
    print(f"Loaded {len(image_files)} images.")

    haar_detection_pipeline(image_files)

if __name__ == "__main__":
    main()