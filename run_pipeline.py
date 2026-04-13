import csv
import os
from datetime import datetime

import numpy as np
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


def log_experiment(metrics, run_id, timestamp, csv_path="experiment_tracker.csv"):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run_id", "date", "f1", "precision", "recall"])
        writer.writerow(
            [
                run_id,
                timestamp,
                f"{metrics['f1']:.2f}",
                f"{metrics['precision']:.2f}",
                f"{metrics['recall']:.2f}",
            ]
        )


def main():
    print(f"🚀 Running: {CONFIG['run_id']}")
    run_date = datetime.now().strftime("%Y-%m-%d")

    data_loader = VQADatasetLoader(CONFIG["dataset_dir"])
    val_dataset = data_loader.load_and_balance(split="val", force_balance=False)

    stage1 = Stage1Generator(model_id=CONFIG["model_s1_path"])
    stage2 = Stage2Segmenter(model_id=CONFIG["model_s2_path"])

    y_true = []
    y_pred = []

    for idx, item in enumerate(val_dataset):
        img_path = item["resolved_image_path"]
        question = item["question"]

        try:
            count, labels = stage1.generate_grounding_plan(img_path, question)
            masks, _ = stage2.generate_masks(img_path, labels)
            pred, max_iou = classify_from_masks(masks, threshold=CONFIG["iou_threshold"])

            binary_label = str(item.get("binary_label", "")).strip().lower()
            if binary_label == "single":
                gt = 1
            elif binary_label == "multiple":
                gt = 0
            else:
                print(f"[{idx + 1}/{len(val_dataset)}] skipped: invalid binary_label={item.get('binary_label')}")
                continue
            y_true.append(gt)
            y_pred.append(pred)

            print(
                f"[{idx + 1}/{len(val_dataset)}] count={count} labels={labels} "
                f"max_iou={max_iou:.3f} pred={'Single' if pred == 1 else 'Multiple'}"
            )
        except Exception as e:
            print(f"[{idx + 1}/{len(val_dataset)}] failed: {e}")
            continue

    if not y_true:
        print("No successful samples.")
        return

    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
    }

    print(
        "✅ Metrics | "
        f"F1: {metrics['f1']:.2f}% | "
        f"Precision: {metrics['precision']:.2f}% | "
        f"Recall: {metrics['recall']:.2f}%"
    )

    log_experiment(metrics, CONFIG["run_id"], run_date)


if __name__ == "__main__":
    main()
