import numpy as np
import cv2
import math

class TopologicalEvaluator:
    # Removed w3_variance. D_Score is now purely Overlap + Global Scatter.
    def __init__(self, w1_ciou=0.4, w2_conflict=0.6, threshold=0.5):
        print("🚀 Initializing Pure Topological Evaluator (IoM-Enhanced)...")
        
        total_weight = w1_ciou + w2_conflict
        self.W1 = w1_ciou / total_weight
        self.W2 = w2_conflict / total_weight
        
        self.threshold = threshold

    def _get_bounding_box(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None 
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return [xmin, ymin, xmax, ymax]

    def _calculate_ciou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        min_area = float(min(boxA_area, boxB_area))
        
        if min_area == 0:
            return 0.0
            
        iom = inter_area / min_area

        # --- TRUE CONTAINMENT OVERRIDE ---
        # If the smaller box is functionally contained inside the larger box,
        # they represent a part-to-whole relationship (Single). 
        # We short-circuit to 1.0 to completely bypass Centroid/Aspect penalties.
        if iom >= 0.85:
            return 1.0
        # ---------------------------------

        center_Ax, center_Ay = (boxA[0] + boxA[2])/2, (boxA[1] + boxA[3])/2
        center_Bx, center_By = (boxB[0] + boxB[2])/2, (boxB[1] + boxB[3])/2
        centroid_distance_sq = (center_Ax - center_Bx)**2 + (center_Ay - center_By)**2
        
        enc_x1, enc_y1 = min(boxA[0], boxB[0]), min(boxA[1], boxB[1])
        enc_x2, enc_y2 = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])
        diagonal_length_sq = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2

        w_A, h_A = boxA[2] - boxA[0], boxA[3] - boxA[1]
        w_B, h_B = boxB[2] - boxB[0], boxB[3] - boxB[1]
        
        v = (4.0 / (math.pi ** 2)) * (math.atan2(w_A, h_A) - math.atan2(w_B, h_B)) ** 2
        
        alpha = v / (1.0 - iom + v + 1e-6)
        
        if diagonal_length_sq == 0:
            return iom
            
        ciou = iom - (centroid_distance_sq / diagonal_length_sq) - (alpha * v)
        return max(0.0, min(1.0, ciou)) 

    def _calculate_conflict_ratio(self, masks):
        all_contour_points = []
        
        for mask in masks:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                points = np.vstack(contours)
                all_contour_points.append(points)

        if not all_contour_points:
            return 0.0
            
        global_points = np.vstack(all_contour_points)
        global_hull = cv2.convexHull(global_points)
        global_hull_area = cv2.contourArea(global_hull)
        
        global_mask_union = np.logical_or.reduce(masks)
        actual_mask_area = np.sum(global_mask_union)
        
        if global_hull_area == 0:
            conflict_ratio = 0.0
        else:
            conflict_ratio = (global_hull_area - actual_mask_area) / global_hull_area
            
        return max(0.0, min(1.0, conflict_ratio))

    def evaluate(self, masks):
        if len(masks) < 2:
            return 1, 0.0  
            
        valid_masks = [m for m in masks if np.any(m)]
        num_masks = len(valid_masks)
        
        if num_masks < 2:
            return 1, 0.0

        ciou_sum = 0.0
        pairs = 0
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                box1 = self._get_bounding_box(valid_masks[i])
                box2 = self._get_bounding_box(valid_masks[j])
                ciou_sum += self._calculate_ciou(box1, box2)
                pairs += 1
                
        avg_ciou = ciou_sum / pairs if pairs > 0 else 0.0
        
        conflict_ratio = self._calculate_conflict_ratio(valid_masks)
        
        # D_SCORE NOW ONLY WEIGHS TRUE SEPARATION AND GLOBAL SCATTER
        d_score = (self.W1 * (1.0 - avg_ciou)) + (self.W2 * conflict_ratio)
        
        classification = 0 if d_score >= self.threshold else 1
        
        return classification, d_score