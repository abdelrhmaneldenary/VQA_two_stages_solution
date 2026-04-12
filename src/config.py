import os

# 1. Environment Detection Router
# If this Kaggle-specific folder exists, we know we are in the cloud.
IS_KAGGLE = os.path.exists("/kaggle/input")

# 2. Dynamic Paths
BASE_DATA_DIR = "/kaggle/input/datasets/abdelrhmanshaheen/answer-therapy" if IS_KAGGLE else "./data"
LOG_FILE_PATH = "experiment_tracker.csv"

# 3. Master Configuration Dictionary
CONFIG = {
    "run_id": "EXP_001",
    "notes": "Full Artifact Generation Run with Modular Config",
    
    # Stage 1 Parameters
    "prompt_strategy": "3-Shot_In_Context",
    "num_beams": 4,
    "lambda_penalty": 0.5,
    
    # Latent Bridge Parameters
    "logit_scale": 1.0,
    
    # Stage 3 Parameters
    "w1_ciou": 0.4,
    "w2_conflict": 0.6,
    "threshold": 0.5, # Default. Will be updated by calibrate_threshold.py
    
    # Paths
    "dataset_dir": BASE_DATA_DIR,
    "log_file": LOG_FILE_PATH
}