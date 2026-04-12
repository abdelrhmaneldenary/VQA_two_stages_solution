import os
import sys
import gc
import csv
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc

sys.path.append(os.getcwd()) # Kaggle module fix
# Import our custom modular architecture
from src.data_loader import VQADatasetLoader
from src.stage_1_generator import Stage1Generator
from src.latent_bridge import LatentBridge
from src.stage_2_segmentor import Stage2Segmentor
from src.stage_3_topology import TopologicalEvaluator
from src.config import CONFIG

# ==========================================
# Artifact & Visualization Helpers
# ==========================================
def create_artifact_dirs(run_id):
    """Creates the folder structure for thesis visualizations."""
    base_dir = f"artifacts/{run_id}"
    dirs = [base_dir, f"{base_dir}/pr_curves", f"{base_dir}/heatmaps", f"{base_dir}/topology_wins"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def save_pr_curve(y_true, y_scores, run_id, base_dir):
    """Plots and saves the Precision-Recall AUC curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Detecting the Single Class)')
    plt.ylabel('Precision (Accuracy of Single Class)')
    plt.title(f'Precision-Recall Curve - {run_id}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{base_dir}/pr_curves/pr_auc_curve.png", dpi=300)
    plt.close()
    return pr_auc

def save_topology_visualization(image_path, masks, d_score, run_id, img_id, base_dir):
    """Draws the SAM 3 Polygons vs Standard Bounding Boxes to prove the math."""
    img = cv2.imread(image_path)
    if img is None: return
    
    img_boxes = img.copy()
    img_polys = img.copy()
    
    for mask in masks:
        if not np.any(mask): continue
        # 1. Simulate the Baseline Failure (Bounding Boxes)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cv2.rectangle(img_boxes, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3) # Red boxes
        
        # 2. Show Our Success (SAM 3 Polygons & Convex Hull)
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_polys, contours, -1, (0, 255, 0), 3) # Green Polygons
        
        if contours:
            points = np.vstack(contours)
            hull = cv2.convexHull(points)
            cv2.drawContours(img_polys, [hull], -1, (255, 255, 0), 2) # Cyan Hulls
            
    # Concatenate side-by-side
    combined = np.hstack((img_boxes, img_polys))
    cv2.putText(combined, f"Baseline (Boxes) vs Ours (Polys) | D_Score: {d_score:.2f}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(f"{base_dir}/topology_wins/eval_{img_id}.png", combined)

# ==========================================
# Metrics & Logging
# ==========================================
def calculate_stratified_metrics(y_true, y_pred, sources, run_id, pr_auc):
    """Calculates exactly what your teammate calculated, plus our new metrics."""
    metrics = {}
    
    # 1. Overall Metrics
    metrics['overall_f1'] = f1_score(y_true, y_pred, zero_division=0) * 100
    metrics['overall_precision'] = precision_score(y_true, y_pred, zero_division=0) * 100
    metrics['overall_recall'] = recall_score(y_true, y_pred, zero_division=0) * 100
    
    # 2. Minority/Majority Splits (Proves Stage 1 & 2)
    metrics['single_recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    metrics['multiple_recall'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    
    # 3. Domain Slices (VQA vs VizWiz)
    for domain in ["VQA", "VizWiz"]:
        d_true = [t for t, s in zip(y_true, sources) if s == domain]
        d_pred = [p for p, s in zip(y_pred, sources) if s == domain]
        
        if len(d_true) > 0:
            metrics[f'{domain.lower()}_f1'] = f1_score(d_true, d_pred, zero_division=0) * 100
            metrics[f'{domain.lower()}_precision'] = precision_score(d_true, d_pred, zero_division=0) * 100
            metrics[f'{domain.lower()}_recall'] = recall_score(d_true, d_pred, zero_division=0) * 100
        else:
            metrics[f'{domain.lower()}_f1'] = metrics[f'{domain.lower()}_precision'] = metrics[f'{domain.lower()}_recall'] = 0.0

    print("\\n" + "="*50)
    print(f"🏆 PIPELINE METRICS: {run_id} 🏆")
    print("="*50)
    print(f"OVERALL   -> F1: {metrics['overall_f1']:.2f}% | Prec: {metrics['overall_precision']:.2f}% | Rec: {metrics['overall_recall']:.2f}%")
    print(f"VQA/COCO  -> F1: {metrics['vqa_f1']:.2f}% | Prec: {metrics['vqa_precision']:.2f}% | Rec: {metrics['vqa_recall']:.2f}%")
    print(f"VIZWIZ    -> F1: {metrics['vizwiz_f1']:.2f}% | Prec: {metrics['vizwiz_precision']:.2f}% | Rec: {metrics['vizwiz_recall']:.2f}%")
    print(f"RECALL    -> Single (Maj): {metrics['single_recall']:.2f}% | Multiple (Min): {metrics['multiple_recall']:.2f}%")
    print(f"PR-AUC    -> {pr_auc:.4f}")
    print("="*50)
    
    return metrics

def log_experiment(run_id, notes, config, metrics, pr_auc, log_file="experiment_tracker.csv"):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Run_ID", "Date", "Notes", "Num_Beams", "Lambda", "W1_CIoU", "W2_Conflict", 
                "Overall_F1", "Overall_Prec", "Overall_Rec", 
                "VQAv2_F1", "VQAv2_Prec", "VQAv2_Rec", 
                "VizWiz_F1", "VizWiz_Prec", "VizWiz_Rec",
                "Single_Recall", "Multiple_Recall", "PR_AUC"
            ])
            
        writer.writerow([
            run_id, datetime.now().strftime("%Y-%m-%d"), notes,
            config['num_beams'], config['lambda_penalty'], config['w1_ciou'], config['w2_conflict'],
            f"{metrics['overall_f1']:.2f}", f"{metrics['overall_precision']:.2f}", f"{metrics['overall_recall']:.2f}",
            f"{metrics['vqa_f1']:.2f}", f"{metrics['vqa_precision']:.2f}", f"{metrics['vqa_recall']:.2f}",
            f"{metrics['vizwiz_f1']:.2f}", f"{metrics['vizwiz_precision']:.2f}", f"{metrics['vizwiz_recall']:.2f}",
            f"{metrics['single_recall']:.2f}", f"{metrics['multiple_recall']:.2f}", f"{pr_auc:.4f}"
        ])

# ==========================================
# The Main Orchestrator
# ==========================================
def main():

    print(f"🚀 Initializing Run: {CONFIG['run_id']}")
    artifact_dir = create_artifact_dirs(CONFIG["run_id"])
    
    data_loader = VQADatasetLoader(CONFIG["dataset_dir"])
    val_dataset = data_loader.load_and_balance(split="val", force_balance=False)
    few_shot_context = data_loader.get_few_shot_context(n_shots=3)

    stage1 = Stage1Generator(model_id=CONFIG["model_s1_path"])  
    bridge = LatentBridge(logit_scale_factor=CONFIG["logit_scale"])    
    stage2 = Stage2Segmentor(model_id=CONFIG["model_s2_path"])
    stage3 = TopologicalEvaluator(w1_ciou=CONFIG["w1_ciou"], w2_conflict=CONFIG["w2_conflict"], threshold=CONFIG["threshold"])

    y_true, y_pred, y_scores, sources = [], [], [], []
    saved_viz_count = 0
    
    test_subset = val_dataset[:50] # Remove slicing for full dataset
    
    for idx, item in enumerate(test_subset):
        print(f"Processing Image {idx+1}/{len(test_subset)}: {item['question']}")
        
        try:
            # 1. Semantics
            candidates, qwen_outputs = stage1.generate_candidates(
                item["resolved_image_path"], item["question"], few_shot_context, 
                CONFIG["num_beams"], CONFIG["lambda_penalty"]
            )
            
            # 2. Bridge
            bimodal_tuples = bridge.process_bimodal_tuples(
                candidates, qwen_outputs, image_token_start=255, image_token_end=511, visual_grid_size=(16, 16)
            )

            # 3. Geometry
            final_masks, _ = stage2.generate_masks(item["resolved_image_path"], bimodal_tuples)

            # 4. Topology
            prediction, d_score = stage3.evaluate(final_masks)
            ground_truth = 1 if item.get("binary_label") == "single" else 0
            
            # Record Data
            y_pred.append(prediction)
            y_true.append(ground_truth)
            sources.append(item.get("source", "Unknown"))
            
            # To plot PR-AUC for the "Single" class (Label 1), we need confidence it is Single.
            # d_score measures Divergence (Multiple). So Single Confidence = 1.0 - d_score.
            y_scores.append(1.0 - d_score)
            
            # Save 5 topological visualizations where D_score proved divergence
            if d_score > 0.4 and saved_viz_count < 5:
                save_topology_visualization(item["resolved_image_path"], final_masks, d_score, CONFIG["run_id"], idx, artifact_dir)
                saved_viz_count += 1

        except Exception as e:
            print(f"  -> ❌ Pipeline failed: {str(e)}")
        
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Final Evaluation & Artifact Generation
    if len(y_pred) > 0:
        pr_auc = save_pr_curve(y_true, y_scores, CONFIG["run_id"], artifact_dir)
        metrics = calculate_stratified_metrics(y_true, y_pred, sources, CONFIG["run_id"], pr_auc)
        log_experiment(CONFIG["run_id"], CONFIG["notes"], CONFIG, metrics, pr_auc)
        print(f"✅ All artifacts saved to /artifacts/{CONFIG['run_id']}/")

if __name__ == "__main__":
    main()