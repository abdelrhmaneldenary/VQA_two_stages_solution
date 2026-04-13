import os
import gc
import json
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import CONFIG
from src.data_loader import VQADatasetLoader
from src.stage_1_generator import Stage1Generator
from src.stage_2_segmenter import Stage2Segmenter

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

def main():
    print(f"🚀 Running ARMORED SIMPLE PIPELINE: {CONFIG['run_id']}")

    data_loader = VQADatasetLoader(CONFIG["dataset_dir"])
    val_dataset = data_loader.load_and_balance(split="val", force_balance=False)

    stage1 = Stage1Generator(model_id=CONFIG["model_s1_path"])
    stage2 = Stage2Segmenter(model_id=CONFIG["model_s2_path"])

    results = {
        "overall": {"y_true": [], "y_pred": []},
        "vqav2": {"y_true": [], "y_pred": []},
        "vizwiz": {"y_true": [], "y_pred": []}
    }

    for idx, item in enumerate(val_dataset):
        img_path = item["resolved_image_path"]
        question = item["question"]
        
        dataset_source = "vizwiz" if "vizwiz" in str(img_path).lower() else "vqav2"

        try:
            # --- MODEL PING-PONG: STAGE 1 ---
            # Move SAM off GPU, Move Qwen to GPU
            stage2.model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            stage1.model.to("cuda:0")

            count, labels = stage1.generate_grounding_plan(img_path, question)

            # --- MODEL PING-PONG: STAGE 2 ---
            # Move Qwen off GPU, Move SAM to GPU
            stage1.model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            stage2.model.to("cuda:0")

            masks, _ = stage2.generate_masks(img_path, labels)
            
            # --- CLASSIFY ---
            pred, max_iou = classify_from_masks(masks, threshold=CONFIG["iou_threshold"])

            binary_label = str(item.get("binary_label", "")).strip().lower()
            if binary_label == "single":
                gt = 1
            elif binary_label == "multiple":
                gt = 0
            else:
                continue

            results["overall"]["y_true"].append(gt)
            results["overall"]["y_pred"].append(pred)
            results[dataset_source]["y_true"].append(gt)
            results[dataset_source]["y_pred"].append(pred)

            print(
                f"[{idx + 1}/{len(val_dataset)}] [{dataset_source.upper()}] "
                f"labels={labels} max_iou={max_iou:.3f} pred={'Single' if pred == 1 else 'Multiple'}"
            )
            
        except Exception as e:
            print(f"[{idx + 1}/{len(val_dataset)}] failed: {e}")
            continue
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # --- METRICS LOGGING ---
    metrics = {}
    for subset in ["overall", "vqav2", "vizwiz"]:
        yt, yp = results[subset]["y_true"], results[subset]["y_pred"]
        if len(yt) > 0:
            metrics[f"{subset}_f1" if subset != "overall" else "overall_f1"] = round(f1_score(yt, yp, zero_division=0) * 100, 2)
            prefix = "vqa" if subset == "vqav2" else subset
            metrics[f"{prefix}_precision"] = round(precision_score(yt, yp, zero_division=0) * 100, 2)
            metrics[f"{prefix}_recall"] = round(recall_score(yt, yp, zero_division=0) * 100, 2)
        else:
             metrics[f"{subset}_f1" if subset != "overall" else "overall_f1"] = 0.0
             prefix = "vqa" if subset == "vqav2" else subset
             metrics[f"{prefix}_precision"] = 0.0
             metrics[f"{prefix}_recall"] = 0.0

    print("\n✅ Final Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()