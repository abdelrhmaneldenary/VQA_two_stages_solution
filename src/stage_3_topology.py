import numpy as np
import cv2

class TopologicalEvaluator:
    def __init__(self, w1_ciou=0.4, w2_conflict=0.6, threshold=0.045):
        print("🚀 Initializing VizWiz Hybrid Topological Evaluator (BBox NMS + Pixel Scatter)...")
        total_weight = w1_ciou + w2_conflict
        self.W1 = w1_ciou / total_weight
        self.W2 = w2_conflict / total_weight
        self.threshold = threshold

    def _get_bounding_box(self, mask):
        """Extracts the outermost bounding box of a pixel mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None 
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return [xmin, ymin, xmax, ymax]

    def _calculate_pixel_iom(self, maskA, maskB):
        """Calculates pixel-perfect Intersection over Minimum."""
        mA = maskA > 0
        mB = maskB > 0
        intersection = np.logical_and(mA, mB).sum()
        min_area = min(mA.sum(), mB.sum())
        if min_area == 0: return 0.0
        return intersection / min_area

    def _calculate_conflict_ratio(self, masks):
        """Calculates macro-scatter using Sklansky's convex hull on pure pixels."""
        all_contour_points = []
        for mask in masks:
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                points = np.vstack(contours)
                all_contour_points.append(points)

        if not all_contour_points: return 0.0
            
        global_points = np.vstack(all_contour_points)
        global_hull = cv2.convexHull(global_points)
        global_hull_area = cv2.contourArea(global_hull)
        
        global_mask_union = np.logical_or.reduce([m > 0 for m in masks])
        actual_mask_area = np.sum(global_mask_union)
        
        if global_hull_area == 0: return 0.0
        conflict_ratio = (global_hull_area - actual_mask_area) / global_hull_area
        return max(0.0, min(1.0, conflict_ratio))

    def evaluate(self, masks):
        if len(masks) < 2:
            return 1, 0.0  
            
        valid_masks = [m for m in masks if np.any(m)]
        if len(valid_masks) < 2:
            return 1, 0.0

        # --- HYBRID SEMANTIC NMS (Using Bounding Boxes) ---
        # This safely absorbs part-to-whole elements (pillows into couches) 
        # without accidentally fusing distinct adjacent objects (two parked cars).
        bboxes = [self._get_bounding_box(m) for m in valid_masks]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) if b else 0 for b in bboxes]
        
        sorted_indices = np.argsort(areas)[::-1]
        kept_indices = []
        
        for i in sorted_indices:
            is_absorbed = False
            box_child = bboxes[i]
            area_child = areas[i]
            
            if area_child == 0: continue
            
            for j in kept_indices:
                box_parent = bboxes[j]
                
                xA = max(box_child[0], box_parent[0])
                yA = max(box_child[1], box_parent[1])
                xB = min(box_child[2], box_parent[2])
                yB = min(box_child[3], box_parent[3])

                inter_area = max(0, xB - xA) * max(0, yB - yA)
                
                # If 85% of the child's bounding box is inside the parent's bounding box
                if (inter_area / area_child) >= 0.85:
                    is_absorbed = True
                    break
                    
            if not is_absorbed:
                kept_indices.append(i)
                
        final_masks = [valid_masks[idx] for idx in kept_indices]
        # --------------------------------------------------

        num_masks = len(final_masks)
        if num_masks < 2:
            return 1, 0.0

        # --- PIXEL DIVERGENCE MATH ---
        # Any remaining masks are truly distinct objects. Evaluate their pixel-perfect scatter.
        overlap_sum = 0.0
        pairs = 0
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                iom = self._calculate_pixel_iom(final_masks[i], final_masks[j])
                overlap_sum += iom
                pairs += 1
                
        avg_overlap = overlap_sum / pairs if pairs > 0 else 0.0
        conflict_ratio = self._calculate_conflict_ratio(final_masks)
        
        d_score = (self.W1 * (1.0 - avg_overlap)) + (self.W2 * conflict_ratio)
        
        classification = 0 if d_score >= self.threshold else 1
        return classification, d_score