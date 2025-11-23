# Data location
RAW_DATA_PATH = "data/"

# Data split parameters
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.1
DATA_SPLIT_TRAIN_PATH = "data/train.txt"
DATA_SPLIT_VAL_PATH = "data/validation.txt"
DATA_SPLIT_TEST_PATH = "data/test.txt"
SEED = 42

# YOLO data parameters
DATA_YOLO_PATH = "data/"

# Model training configuration
MODEL = "yolov8m.pt"
DATA = "data.yaml"
EPOCHS = 90
PATIENCE = 20
BATCH = 8
WORKERS = 16
IMGSZ = 1280
CLOSE_MOSAIC = 15

