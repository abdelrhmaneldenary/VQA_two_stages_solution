import numpy as np
import cv2
import math

class TopologicalEvaluator:
    def __init__(self, w1_ciou=0.4, w2_conflict=0.6, threshold=0.5):
        print("🚀 Initializing VizWiz Pixel-Perfect Topological Evaluator...")
        total_weight = w1_ciou + w2_conflict
        self.W1 = w1_ciou / total_weight
        self.W2 = w2_conflict / total_weight
        self.threshold = threshold

    def _calculate_pixel_iom(self, maskA, maskB):
        """
        Calculates Intersection over Minimum (Containment) using pure pixels.
        Matches the tight polygon logic of the VizWiz challenge.
        """
        # Ensure masks are boolean
        mA = maskA > 0
        mB = maskB > 0
        
        intersection = np.logical_and(mA, mB).sum()
        areaA = mA.sum()
        areaB = mB.sum()
        
        min_area = min(areaA, areaB)
        if min_area == 0:
            return 0.0
            
        return intersection / min_area

    def _calculate_conflict_ratio(self, masks):
        all_contour_points = []
        for mask in masks:
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                points = np.vstack(contours)
                all_contour_points.append(points)

        if not all_contour_points:
            return 0.0
            
        global_points = np.vstack(all_contour_points)
        global_hull = cv2.convexHull(global_points)
        global_hull_area = cv2.contourArea(global_hull)
        
        global_mask_union = np.logical_or.reduce([m > 0 for m in masks])
        actual_mask_area = np.sum(global_mask_union)
        
        if global_hull_area == 0:
            return 0.0
            
        conflict_ratio = (global_hull_area - actual_mask_area) / global_hull_area
        return max(0.0, min(1.0, conflict_ratio))

    def evaluate(self, masks):
        if len(masks) < 2:
            return 1, 0.0  
            
        valid_masks = [m for m in masks if np.any(m)]
        if len(valid_masks) < 2:
            return 1, 0.0

        # --- PIXEL-PERFECT SEMANTIC NMS ---
        # Sort masks by pure pixel area (Largest to Smallest)
        areas = [np.sum(m > 0) for m in valid_masks]
        sorted_indices = np.argsort(areas)[::-1]
        kept_indices = []
        
        for i in sorted_indices:
            is_absorbed = False
            mask_child = valid_masks[i]
            
            for j in kept_indices:
                mask_parent = valid_masks[j]
                
                # Check pure pixel containment
                iom = self._calculate_pixel_iom(mask_parent, mask_child)
                
                # If 85% of the child's pixels are inside the parent's pixels, absorb it.
                if iom >= 0.85:
                    is_absorbed = True
                    break
                    
            if not is_absorbed:
                kept_indices.append(i)
                
        final_masks = [valid_masks[idx] for idx in kept_indices]
        # ----------------------------------

        num_masks = len(final_masks)
        if num_masks < 2:
            return 1, 0.0

        # --- DIVERGENCE MATH ---
        # If masks survived absorption, they are distinct objects.
        # We calculate how much they still overlap (if at all).
        overlap_sum = 0.0
        pairs = 0
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                iom = self._calculate_pixel_iom(final_masks[i], final_masks[j])
                overlap_sum += iom
                pairs += 1
                
        avg_overlap = overlap_sum / pairs if pairs > 0 else 0.0
        conflict_ratio = self._calculate_conflict_ratio(final_masks)
        
        # D_Score = (Weight of Non-Overlap) + (Weight of Global Scatter)
        d_score = (self.W1 * (1.0 - avg_overlap)) + (self.W2 * conflict_ratio)
        
        classification = 0 if d_score >= self.threshold else 1
        return classification, d_score