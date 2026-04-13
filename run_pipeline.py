import csv
import os
import json
import gc
import torch
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from PIL import Image

from src.config import CONFIG
from src.data_loader import VQADatasetLoader
from src.stage_1_generator import Stage1Generator
from src.stage_2_segmenter import Stage2Segmenter

# (Assuming resize_for_vram is defined here or imported)
def resize_for_vram(image_path, max_dim=1024):
    """Prevents 40GB VRAM explosions by capping image resolution."""
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

def calculate_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(a, b).sum()
    return intersection / union

def classify_from_masks(masks, threshold=0.5):
    valid = [m for m in masks if np.any(m)]
    if len(valid) < 2:
        return 1, 1.0

    ious = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ious.append(calculate_iou(valid[i], valid[j]))

    max_iou = max(ious) if ious else 0.0
    prediction = 1 if max_iou > threshold else 0
    return prediction, max_iou

def log_experiment(metrics, run_id, timestamp, csv_path="experiment_tracker.csv"):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "run_id", "date", 
                "overall_f1", "overall_precision", "overall_recall",
                "vqav2_f1", "vqa_precision", "vqa_recall",
                "vizwiz_f1", "vizwiz_precision", "vizwiz_recall"
            ])
        writer.writerow([
            run_id, timestamp,
            f"{metrics['overall_f1']:.2f}", f"{metrics['overall_precision']:.2f}", f"{metrics['overall_recall']:.2f}",
            f"{metrics['vqav2_f1']:.2f}", f"{metrics['vqa_precision']:.2f}", f"{metrics['vqa_recall']:.2f}",
            f"{metrics['vizwiz_f1']:.2f}", f"{metrics['vizwiz_precision']:.2f}", f"{metrics['vizwiz_recall']:.2f}",
        ])

def main():
    print(f"🚀 Running: {CONFIG['run_id']}")
    run_date = datetime.now().strftime("%Y-%m-%d")

    data_loader = VQADatasetLoader(CONFIG["dataset_dir"])
    val_dataset = data_loader.load_and_balance(split="val", force_balance=False)

    stage1 = Stage1Generator(model_id=CONFIG["model_s1_path"])
    stage2 = Stage2Segmenter(model_id=CONFIG["model_s2_path"])

    # Dictionaries to track metrics separately
    results = {
        "overall": {"y_true": [], "y_pred": []},
        "vqav2": {"y_true": [], "y_pred": []},
        "vizwiz": {"y_true": [], "y_pred": []}
    }

    for idx, item in enumerate(val_dataset):
        img_path = item["resolved_image_path"]
        question = item["question"]

        try:
            image = resize_for_vram(img_path, max_dim=1024)
            count, labels = stage1.generate_grounding_plan(image, question)
            masks, _ = stage2.generate_masks(image, labels)
            pred, max_iou = classify_from_masks(masks, threshold=CONFIG["iou_threshold"])

            binary_label = str(item.get("binary_label", "")).strip().lower()
            if binary_label == "single":
                gt = 1
            elif binary_label == "multiple":
                gt = 0
            else:
                print(f"[{idx + 1}/{len(val_dataset)}] skipped: invalid binary_label={item.get('binary_label')}")
                continue
            
            # Determine dataset source based on path or metadata
            path_str = str(img_path).lower()
            dataset_source = "vizwiz" if "vizwiz" in path_str else "vqav2"

            # Append to overall
            results["overall"]["y_true"].append(gt)
            results["overall"]["y_pred"].append(pred)
            
            # Append to specific split
            results[dataset_source]["y_true"].append(gt)
            results[dataset_source]["y_pred"].append(pred)

            print(
                f"[{idx + 1}/{len(val_dataset)}] [{dataset_source.upper()}] count={count} labels={labels} "
                f"max_iou={max_iou:.3f} pred={'Single' if pred == 1 else 'Multiple'}"
            )
        except Exception as e:
            print(f"[{idx + 1}/{len(val_dataset)}] failed: {e}")
            continue
        finally:
            # Prevent Kaggle T4 OOM Crashes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if not results["overall"]["y_true"]:
        print("No successful samples.")
        return

    # Calculate metrics for all subsets
    metrics = {}
    for subset in ["overall", "vqav2", "vizwiz"]:
        yt = results[subset]["y_true"]
        yp = results[subset]["y_pred"]
        
        # Avoid zero division if a subset is completely missing
        if len(yt) > 0:
            f1 = f1_score(yt, yp, zero_division=0) * 100
            p = precision_score(yt, yp, zero_division=0) * 100
            r = recall_score(yt, yp, zero_division=0) * 100
        else:
            f1, p, r = 0.0, 0.0, 0.0

        # Format exactly as requested
        prefix = "vqa" if subset == "vqav2" else subset
        
        metrics[f"{subset}_f1" if subset != "overall" else "overall_f1"] = round(f1, 2)
        metrics[f"{prefix}_precision"] = round(p, 2)
        metrics[f"{prefix}_recall"] = round(r, 2)

    # Print exact JSON format requested
    print("\n✅ Final Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    log_experiment(metrics, CONFIG["run_id"], run_date)


if __name__ == "__main__":
    main()