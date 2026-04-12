import numpy as np
from sklearn.metrics import f1_score

def find_optimal_threshold(y_true, d_scores):
    """
    Sweeps through thresholds from 0.0 to 1.0 to find the exact cutoff 
    that maximizes the F1 Score on the validation set.
    """
    print("Calibrating Optimal D_Score Threshold via PR-AUC Sweep...")
    
    best_threshold = 0.5
    max_f1 = 0.0
    
    # Test 100 thresholds between 0.01 and 0.99
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for thresh in thresholds:
        # Remember: d_score measures Divergence. 
        # If d_score >= thresh, classify as 0 (Multiple). Else 1 (Single).
        y_pred = [0 if score >= thresh else 1 for score in d_scores]
        
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = thresh
            
    print(f"✅ Calibration Complete.")
    print(f"-> Optimal Threshold: {best_threshold:.3f}")
    print(f"-> Maximum Val F1:    {max_f1 * 100:.2f}%")
    
    return best_threshold