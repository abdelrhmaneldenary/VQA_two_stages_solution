import os
# PyTorch VRAM Fragmentation Override (Must be before torch is imported)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import traceback

sys.path.append(os.getcwd()) # Kaggle module fix
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
    base_dir = f"artifacts/{run_id}"
    dirs = [base_dir, f"{base_dir}/pr_curves", f"{base_dir}/heatmaps", f"{base_dir}/topology_wins"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def save_pr_curve(y_true, y_scores, run_id, base_dir):
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
    img = cv2.imread(image_path)
    if img is None: return
    
    img_boxes = img.copy()
    img_polys = img.copy()
    
    for mask in masks:
        if not np.any(mask): continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cv2.rectangle(img_boxes, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3) 
        
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_polys, contours, -1, (0, 255, 0), 3) 
        
        if contours:
            points = np.vstack(contours)
            hull = cv2.convexHull(points)
            cv2.drawContours(img_polys, [hull], -1, (255, 255, 0), 2) 
            
    combined = np.hstack((img_boxes, img_polys))
    cv2.putText(combined, f"Baseline vs Ours | D_Score: {d_score:.2f}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(f"{base_dir}/topology_wins/eval_{img_id}.png", combined)

# ==========================================
# Metrics & Logging
# ==========================================
def calculate_stratified_metrics(y_true, y_pred, sources, run_id, pr_auc):
    metrics = {}
    metrics['overall_f1'] = f1_score(y_true, y_pred, zero_division=0) * 100
    metrics['overall_precision'] = precision_score(y_true, y_pred, zero_division=0) * 100
    metrics['overall_recall'] = recall_score(y_true, y_pred, zero_division=0) * 100
    metrics['single_recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    metrics['multiple_recall'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    
    for domain in ["VQA", "VizWiz"]:
        d_true = [t for t, s in zip(y_true, sources) if s == domain]
        d_pred = [p for p, s in zip(y_pred, sources) if s == domain]
        if len(d_true) > 0:
            metrics[f'{domain.lower()}_f1'] = f1_score(d_true, d_pred, zero_division=0) * 100
            metrics[f'{domain.lower()}_precision'] = precision_score(d_true, d_pred, zero_division=0) * 100
            metrics[f'{domain.lower()}_recall'] = recall_score(d_true, d_pred, zero_division=0) * 100
        else:
            metrics[f'{domain.lower()}_f1'] = metrics[f'{domain.lower()}_precision'] = metrics[f'{domain.lower()}_recall'] = 0.0

    print("\n" + "="*50)
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
import os
# PyTorch VRAM Fragmentation Override (Must be before torch is imported)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import traceback
from PIL import Image  # <--- REQUIRED FOR THE AFFINE MATH

sys.path.append(os.getcwd()) # Kaggle module fix
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
    base_dir = f"artifacts/{run_id}"
    dirs = [base_dir, f"{base_dir}/pr_curves", f"{base_dir}/heatmaps", f"{base_dir}/topology_wins"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base_dir

def save_pr_curve(y_true, y_scores, run_id, base_dir):
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
    img = cv2.imread(image_path)
    if img is None: return
    
    img_boxes = img.copy()
    img_polys = img.copy()
    
    for mask in masks:
        if not np.any(mask): continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cv2.rectangle(img_boxes, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3) 
        
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_polys, contours, -1, (0, 255, 0), 3) 
        
        if contours:
            points = np.vstack(contours)
            hull = cv2.convexHull(points)
            cv2.drawContours(img_polys, [hull], -1, (255, 255, 0), 2) 
            
    combined = np.hstack((img_boxes, img_polys))
    cv2.putText(combined, f"Baseline vs Ours | D_Score: {d_score:.2f}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(f"{base_dir}/topology_wins/eval_{img_id}.png", combined)

# ==========================================
# Metrics & Logging
# ==========================================
def calculate_stratified_metrics(y_true, y_pred, sources, run_id, pr_auc):
    metrics = {}
    metrics['overall_f1'] = f1_score(y_true, y_pred, zero_division=0) * 100
    metrics['overall_precision'] = precision_score(y_true, y_pred, zero_division=0) * 100
    metrics['overall_recall'] = recall_score(y_true, y_pred, zero_division=0) * 100
    metrics['single_recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    metrics['multiple_recall'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    
    for domain in ["VQA", "VizWiz"]:
        d_true = [t for t, s in zip(y_true, sources) if s == domain]
        d_pred = [p for p, s in zip(y_pred, sources) if s == domain]
        if len(d_true) > 0:
            metrics[f'{domain.lower()}_f1'] = f1_score(d_true, d_pred, zero_division=0) * 100
            metrics[f'{domain.lower()}_precision'] = precision_score(d_true, d_pred, zero_division=0) * 100
            metrics[f'{domain.lower()}_recall'] = recall_score(d_true, d_pred, zero_division=0) * 100
        else:
            metrics[f'{domain.lower()}_f1'] = metrics[f'{domain.lower()}_precision'] = metrics[f'{domain.lower()}_recall'] = 0.0

    print("\n" + "="*50)
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
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

def debug_visualize(image_path, bimodal_tuples, masks, idx, base_dir):
    """
    Renders a 1x2 diagnostic plot.
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left: Qwen Anchor Points
    ax[0].imshow(img)
    colors = ['red', 'blue', 'green', 'yellow', 'magenta']
    for i, (text, pt) in enumerate(bimodal_tuples):
        c = colors[i % len(colors)]
        ax[0].scatter(pt[0], pt[1], s=200, color=c, edgecolors='white', label=f'{text}')
        ax[0].text(pt[0]+10, pt[1]+10, text, color='white', fontsize=12, backgroundcolor='black')
    ax[0].set_title(f"Image {idx}: Qwen Anchor Points")
    
    # Right: SAM 3 Masks
    ax[1].imshow(img)
    mask_found = False
    for i, mask in enumerate(masks):
        if np.any(mask):
            mask_found = True
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask > 0] = [0, 1, 0, 0.4] # Green masks
            ax[1].imshow(overlay)
    
    ax[1].set_title("SAM 3: Masks" if mask_found else "SAM 3: EMPTY")
    
    plt.tight_layout()
    save_path = f"{base_dir}/debug_{idx}.png"
    plt.savefig(save_path)
    plt.show() # <--- This will show the image in your Kaggle Cell!
    plt.close()

# ==========================================
# The Main Orchestrator
# ==========================================
def main():
    print(f"🚀 Initializing DEBUG Run: {CONFIG['run_id']}")
    artifact_dir = create_artifact_dirs(CONFIG["run_id"])
    
    data_loader = VQADatasetLoader(CONFIG["dataset_dir"])
    val_dataset = data_loader.load_and_balance(split="val", force_balance=False)
    few_shot_context = " " # Or your loaded context

    # --- Standard Init ---
    stage1 = Stage1Generator(model_id=CONFIG["model_s1_path"])  
    bridge = LatentBridge() # Ensure this matches our updated no-arg Bridge
    stage2 = Stage2Segmentor(model_id=CONFIG["model_s2_path"])
    stage3 = TopologicalEvaluator(w1_ciou=CONFIG["w1_ciou"], w2_conflict=CONFIG["w2_conflict"])

    # Limit to 10 for Real Debugging
# Hunt down 5 images specifically labeled as "multiple"
    test_subset = [item for item in val_dataset if item.get("binary_label") == "multiple"][:5]    
    y_true, y_pred, y_scores, sources = [], [], [], []

    for idx, item in enumerate(test_subset):
        img_path = item["resolved_image_path"]
        print(f"\n[IMAGE {idx+1}/10] Question: {item['question']}")
        
        try:
            # Get dimensions for affine verification
            raw_w, raw_h = Image.open(img_path).size
            print(f"   -> 📐 Native Resolution: {raw_w}x{raw_h}")
            
            # 1. Semantics (Stage 1)
            candidates, qwen_outputs, start_idx, end_idx, grid_h, grid_w = stage1.generate_candidates(
                img_path, item["question"], few_shot_context, num_beams=CONFIG["num_beams"]
            )
            print(f"   -> 🧠 Qwen Grid: {grid_w}w x {grid_h}h (Tokens: {grid_w*grid_h})")
            
            # 2. Bridge (Point Extraction)
            bimodal_tuples = bridge.process_bimodal_tuples(
                candidates, qwen_outputs, start_idx, end_idx, grid_h, grid_w, (raw_w, raw_h)
            )

            # --- COORDINATE LOGGING ---
            for text, pt in bimodal_tuples:
                print(f"   -> 🌉 Bridge Anchor: '{text}' at Pixel (X:{pt[0]}, Y:{pt[1]})")

            # VRAM Safety
            del qwen_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # 3. Geometry (SAM 3)
            final_masks, _ = stage2.generate_masks(img_path, bimodal_tuples)
            
            # --- VISUAL TELEMETRY ---
            debug_visualize(
                image_path=item["resolved_image_path"], 
                bimodal_tuples=bimodal_tuples, 
                masks=final_masks, 
                idx=idx, 
                base_dir=artifact_dir # Matches the 5 arguments now
            )            
            # 4. Topology (D_Score)
            prediction, d_score = stage3.evaluate(final_masks)
            ground_truth = 1 if item.get("binary_label") == "single" else 0
            
            valid_count = sum([1 for m in final_masks if np.any(m)])
            print(f"   -> 📐 Stage 2 Result: {valid_count}/{len(final_masks)} masks contain pixels.")
            print(f"   -> ⚖️ D_Score: {d_score:.3f} | Pred: {prediction} | True: {ground_truth}")

            y_pred.append(prediction)
            y_true.append(ground_truth)
            y_scores.append(1.0 - d_score)
            sources.append(item.get("source", "Unknown"))

        except Exception as e:
            print(f"   -> ❌ CRITICAL FAILURE at Image {idx}: {e}")
            traceback.print_exc()
            break # Stop and inspect the fail

    # Final PR-AUC for the 10-image subset
    if len(y_pred) > 0:
        pr_auc = save_pr_curve(y_true, y_scores, CONFIG["run_id"], artifact_dir)
        print(f"\n✅ Debug Telemetry complete. Check /artifacts/{CONFIG['run_id']} for PNGs.")

if __name__ == "__main__":
    main()