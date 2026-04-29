import os

# ─────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────
YOLOV8_MODEL_PATH  = "/Users/himasri/Downloads/Real-Time-Flood-Evacuation_Optimizer-main/runs/detect/train5/weights/best.pt"
CNN_MODEL_PATH     = "weights/flood_cnn.pth"
BASE_MODEL         = "yolov8n.pt"

# ─────────────────────────────────────────
# INPUT SOURCE
# ─────────────────────────────────────────
INPUT_SOURCE = "flood_test1.mp4"
CAMERA_ID          = "CAM_01"

# 🔥 REDUCED RESOLUTION (for speed)
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 400
CNN_IMAGE_SIZE = 224
CNN_RISK_LABELS = ["SAFE", "LOW", "MEDIUM", "HIGH"]

# ─────────────────────────────────────────
# THRESHOLDS / PARAMETERS
# ─────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.25
IOU_THRESHOLD         = 0.45
RISK_UPDATE_INTERVAL  = 60
HIGH_RISK_THRESHOLD   = 8.0
MEDIUM_RISK_THRESHOLD = 4.0

# ─────────────────────────────────────────
# YOLO CLASSES
# ─────────────────────────────────────────
CLASS_NAMES = {
    0: "flood_water",
    1: "submerged_vehicle",
    2: "stranded_person",
    3: "road_debris",
    4: "water_marker_safe",
    5: "water_marker_danger",
    6: "sos_person",
}

CLASS_RISK_WEIGHTS = {
    "flood_water":          2,
    "submerged_vehicle":    3,
    "stranded_person":      4,
    "road_debris":          2,
    "water_marker_safe":    0,
    "water_marker_danger":  5,
    "sos_person":           5,
}

CLASS_COLORS = {
    "flood_water":          (255, 165, 0),
    "submerged_vehicle":    (0,   0,   255),
    "stranded_person":      (0,   255, 255),
    "road_debris":          (128, 0,   128),
    "water_marker_safe":    (0,   255, 0),
    "water_marker_danger":  (0,   0,   200),
    "sos_person":           (0,   0,   255),
}

# ─────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────
AVG_SPEED_KMPH     = 30
START_NODE         = "A"
GOAL_NODE          = "F"

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
LOG_DIR            = "logs"
OUTPUT_DIR         = "outputs"

OUTPUT_VIDEO_PATH  = os.path.join(OUTPUT_DIR, "output.avi")

# create folders
for d in [LOG_DIR, OUTPUT_DIR, "weights"]:
    os.makedirs(d, exist_ok=True)
# ─────────────────────────────────────────
# LOG PATHS (FIX FOR LOGGER)
# ─────────────────────────────────────────
LOG_DIR            = "logs"
LOG_FILE_PATH      = os.path.join(LOG_DIR, "system.log")
DETECTION_LOG_PATH = os.path.join(LOG_DIR, "detections.csv")
ALERT_LOG_PATH     = os.path.join(LOG_DIR, "alerts.json")
# ─────────────────────────────────────────
# TRAINING SETTINGS (FIX)
# ─────────────────────────────────────────
DATASET_YAML     = "data.yaml"
TRAIN_EPOCHS     = 100
TRAIN_BATCH_SIZE = 16
TRAIN_IMAGE_SIZE = 640
TRAIN_LR         = 0.001
CNN_TRAIN_EPOCHS = 20
CNN_NUM_CLASSES  = 4
