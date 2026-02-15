# ──────────────────────────────────────────────
# Crowd Risk Monitoring — Configuration
# ──────────────────────────────────────────────

# Zone capacity (max safe occupancy for the monitored area)
ZONE_CAPACITY = 1

# Camera
CAMERA_INDEX = 1  # 0 = default webcam

# YOLO
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.4  # ignore detections below this confidence

# Risk engine — EMA smoothing
SMOOTHING_ALPHA = 0.7  # weight for current count (1-alpha for previous)

# Risk engine — growth rate window
GROWTH_RATE_WINDOW = 3  # compare current count against count N readings ago

# Risk engine — persistence / hysteresis
HIGH_DENSITY_THRESHOLD = 0.8    # density_ratio >= this enters high state
HIGH_DENSITY_EXIT = 0.7         # density_ratio < this exits high state (hysteresis)
CRITICAL_PERSISTENCE_SEC = 10   # seconds in high state before Critical

# Risk engine — surge detection
SURGE_GROWTH_THRESHOLD = 3  # persons gained within the growth rate window

# Risk engine — compound risk score weights (must sum to 1.0)
WEIGHT_DENSITY = 0.5
WEIGHT_GROWTH = 0.3
WEIGHT_PERSISTENCE = 0.2

# Risk score classification thresholds
RISK_THRESHOLD_LOW = 0.4
RISK_THRESHOLD_MEDIUM = 0.6
RISK_THRESHOLD_HIGH = 0.8

# Server
FLASK_PORT = 5001

# Database
DATABASE_PATH = "crowd_data.db"
