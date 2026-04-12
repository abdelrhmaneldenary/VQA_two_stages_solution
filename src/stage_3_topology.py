import numpy as np
import cv2
import math

class TopologicalEvaluator:
    def __init__(self, w1_ciou=0.4, w2_conflict=0.4, w3_variance=0.2, threshold=0.5):
        """
        Initializes the topological judge. The weights determine how heavily 
        we penalize local overlap versus global scattering. The threshold 
        will later be dynamically overridden by our PR-AUC calibration.
        """
        print("Initializing Stage 3 Topological Evaluator...")
        self.W1 = w1_ciou
        self.W2 = w2_conflict
        self.W3 = w3_variance
        self.threshold = threshold

    def _get_bounding_box(self, mask):
        """Extracts the [x1, y1, x2, y2] bounding box from a boolean mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None # Empty mask safeguard
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return [xmin, ymin, xmax, ymax]

    def _calculate_ciou(self, boxA, boxB):
        """
        Calculates the Complete-IoU between two bounding boxes.
        Penalizes standard IoU using Centroid Distance and Aspect Ratio.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Standard Intersection Area
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = float(boxA_area + boxB_area - inter_area)
        
        if union_area == 0:
            return 0.0
            
        iou = inter_area / union_area

        # Centroid Distance Calculation
        center_Ax, center_Ay = (boxA[0] + boxA[2])/2, (boxA[1] + boxA[3])/2
        center_Bx, center_By = (boxB[0] + boxB[2])/2, (boxB[1] + boxB[3])/2
        centroid_distance_sq = (center_Ax - center_Bx)**2 + (center_Ay - center_By)**2
        
        # Enclosing Box Calculation (The smallest box that fits both)
        enc_x1, enc_y1 = min(boxA[0], boxB[0]), min(boxA[1], boxB[1])
        enc_x2, enc_y2 = max(boxA[2], boxB[2]), max(boxA[3], boxB[3])
        diagonal_length_sq = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2

        # Aspect Ratio Consistency (The v and alpha parameters from the YOLO paper)
        w_A, h_A = boxA[2] - boxA[0], boxA[3] - boxA[1]
        w_B, h_B = boxB[2] - boxB[0], boxB[3] - boxB[1]
        
        v = (4 / (math.pi ** 2)) * (math.atan(w_A / h_A) - math.atan(w_B / h_B)) ** 2
        alpha = v / (1 - iou + v + 1e-6)
        
        # Final CIoU
        if diagonal_length_sq == 0:
            return iou
            
        ciou = iou - (centroid_distance_sq / diagonal_length_sq) - (alpha * v)
        return max(0.0, min(1.0, ciou)) # Clamp between 0 and 1

    def _calculate_conflict_ratio_and_variance(self, masks):
        """
        Calculates the Global Conflict Ratio (Empty Space) and the 
        Area Variance of individual convex hulls.
        """
        hull_areas = []
        all_contour_points = []
        
        # 1. Process individual masks for Area Variance
        for mask in masks:
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Compute individual convex hull via Sklansky's algorithm
            points = np.vstack(contours)
            hull = cv2.convexHull(points)
            hull_areas.append(cv2.contourArea(hull))
            all_contour_points.append(points)

        # Variance normalization (0 to 1 scaling based on maximum possible area differences)
        if len(hull_areas) < 2:
            variance_norm = 0.0
        else:
            variance = np.var(hull_areas)
            max_area = max(hull_areas)
            variance_norm = variance / (max_area**2 + 1e-6)

        # 2. Process the Global Conflict Region
        if not all_contour_points:
            return 0.0, variance_norm
            
        # Create the overarching "Rubber Band"
        global_points = np.vstack(all_contour_points)
        global_hull = cv2.convexHull(global_points)
        global_hull_area = cv2.contourArea(global_hull)
        
        # Calculate the actual physical union of all SAM 3 pixels
        global_mask_union = np.logical_or.reduce(masks)
        actual_mask_area = np.sum(global_mask_union)
        
        # Conflict Ratio = (Hull Area - Actual Object Area) / Hull Area
        if global_hull_area == 0:
            conflict_ratio = 0.0
        else:
            conflict_ratio = (global_hull_area - actual_mask_area) / global_hull_area
            
        return max(0.0, min(1.0, conflict_ratio)), variance_norm

    def evaluate(self, masks):
        """
        Executes the final $D_{score}$ equation and outputs the binary classification.
        """
        # Edge Case: If Stage 1 only found 1 object, it is inherently Single.
        if len(masks) < 2:
            return 1, 0.0  # Label 1 (Single), D_score 0.0
            
        # 1. Calculate Average CIoU Across all pairs
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
        
        # 2. Calculate Topological Empty Space
        conflict_ratio, variance_norm = self._calculate_conflict_ratio_and_variance(valid_masks)
        
        # 3. The D_score Equation
        # Note: We use (1 - avg_ciou) because a HIGH overlap means LOW divergence.
        d_score = (self.W1 * (1.0 - avg_ciou)) + (self.W2 * conflict_ratio) + (self.W3 * variance_norm)
        
        # Final classification
        classification = 0 if d_score >= self.threshold else 1
        
        return classification, d_score