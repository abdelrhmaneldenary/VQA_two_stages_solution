import os

IS_KAGGLE = os.path.exists("/kaggle/input")
BASE_DATA_DIR = "/kaggle/input/datasets/abdelrhmanshaheen/answer-therapy" if IS_KAGGLE else "./data"

CONFIG = {
    "run_id": "EXP_QWEN25_SAM3_SIMPLE",
    "notes": "Simple Qwen2.5-VL + SAM3 IoU baseline",
    "model_s1_path": "/kaggle/input/datasets/ruhul77/qwen2-vl-2b-instruct",
    "model_s2_path": "/kaggle/input/datasets/abdelrhmaneldenary/sam3-official-weights-v3"  ,
    "iou_threshold": 0.5,
    "dataset_dir": BASE_DATA_DIR,
}
