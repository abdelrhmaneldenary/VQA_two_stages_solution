import numpy as np
import cv2

class TopologicalEvaluator:
    def __init__(self, w1_ciou=0.4, w2_conflict=0.6, w3_anchor=0.5, threshold=0.045):
        print("🚀 Initializing VizWiz Anchor-Aware Topological Evaluator...")
        total_weight = w1_ciou + w2_conflict + w3_anchor
        self.W1 = w1_ciou / total_weight
        self.W2 = w2_conflict / total_weight
        self.W3 = w3_anchor / total_weight
        self.threshold = threshold

    def _fill_holes(self, mask):
        """
        Finds the outermost contour of the mask and fills all internal holes.
        Turns a 'donut' (couch with a pillow hole) into a solid object.
        """
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        # RETR_EXTERNAL explicitly ignores internal holes and only grabs the outer perimeter
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask = np.zeros_like(mask_uint8)
        if contours:
            # thickness=-1 fills the entire polygon solidly
            cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)
            
        return filled_mask > 0

    def _calculate_pixel_iom(self, maskA, maskB):
        """Calculates pixel-perfect Intersection over Minimum."""
        mA = maskA > 0
        mB = maskB > 0
        intersection = np.logical_and(mA, mB).sum()
        min_area = min(mA.sum(), mB.sum())
        if min_area == 0: return 0.0
        return intersection / min_area

    def _calculate_anchor_separation(self, anchor_points, image_size):
        """
        Computes the maximum pairwise normalized distance between anchor points,
        scaled by the image diagonal.  Returns a value in [0, 1] where a higher
        value indicates that the candidate anchors are spread across physically
        distant regions of the image.
        """
        if len(anchor_points) < 2:
            return 0.0
        img_w, img_h = image_size
        diagonal = np.hypot(img_w, img_h)
        if diagonal == 0:
            return 0.0
        max_dist = 0.0
        for i in range(len(anchor_points)):
            for j in range(i + 1, len(anchor_points)):
                ax, ay = anchor_points[i]
                bx, by = anchor_points[j]
                dist = np.hypot(ax - bx, ay - by) / diagonal
                if dist > max_dist:
                    max_dist = dist
        return min(1.0, max_dist)

    def _calculate_conflict_ratio(self, masks):
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

    def evaluate(self, masks, anchor_points=None, image_size=None):
        if len(masks) < 2:
            return 1, 0.0

        # Filter out empty masks while keeping anchor-point correspondence
        valid_indices = [i for i, m in enumerate(masks) if np.any(m)]
        valid_masks = [masks[i] for i in valid_indices]
        valid_anchors = (
            [anchor_points[i] for i in valid_indices]
            if anchor_points is not None and len(anchor_points) == len(masks)
            else None
        )

        if len(valid_masks) < 2:
            return 1, 0.0

        # --- SOLID-PIXEL ANCHOR-AWARE SEMANTIC NMS ---
        # Fill internal holes so that parent objects (e.g. couch) absorb their
        # children (e.g. pillow) correctly.  However, we add a spatial guard:
        # two candidates whose anchor points are far apart represent physically
        # separate entities (e.g. two text labels on a bottle) and must never
        # be absorbed into each other even when SAM happened to return the same
        # enclosing-object mask for both anchor points.
        solid_masks = [self._fill_holes(m) for m in valid_masks]
        areas = [np.sum(m) for m in solid_masks]

        # Pre-compute image diagonal once for the anchor-distance guard
        diagonal = np.hypot(image_size[0], image_size[1]) if image_size is not None else None

        sorted_indices = np.argsort(areas)[::-1]
        kept_indices = []

        for i in sorted_indices:
            is_absorbed = False
            mask_child = solid_masks[i]

            if areas[i] == 0:
                continue

            for j in kept_indices:
                mask_parent = solid_masks[j]

                iom = self._calculate_pixel_iom(mask_parent, mask_child)

                if iom >= 0.85:
                    # Anchor-aware guard: if the two candidates point to
                    # spatially distant regions, they represent distinct
                    # entities (e.g. separate text labels printed on the
                    # same physical object).  Skip absorption in that case.
                    if valid_anchors is not None and diagonal is not None and diagonal > 0:
                        ax, ay = valid_anchors[i]
                        bx, by = valid_anchors[j]
                        sep = np.hypot(ax - bx, ay - by) / diagonal
                        if sep > 0.15:
                            continue  # Spatially separate — do NOT absorb

                    is_absorbed = True
                    break

            if not is_absorbed:
                kept_indices.append(i)

        # Retrieve the original VizWiz-accurate masks and their anchor points
        # for the survivors
        final_masks = [valid_masks[idx] for idx in kept_indices]
        final_anchors = (
            [valid_anchors[idx] for idx in kept_indices]
            if valid_anchors is not None
            else None
        )
        # -------------------------------------------------------

        num_masks = len(final_masks)
        if num_masks < 2:
            return 1, 0.0

        # --- PIXEL DIVERGENCE MATH ---
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

        # Anchor-separation boost: surviving candidates whose spatial anchors
        # are spread far apart signal that they represent genuinely distinct
        # scene elements even when their SAM masks have high pixel overlap
        # (the classic OCR / text-label failure mode).
        if final_anchors is not None and image_size is not None and len(final_anchors) >= 2:
            anchor_sep = self._calculate_anchor_separation(final_anchors, image_size)
            d_score += self.W3 * anchor_sep

        classification = 0 if d_score >= self.threshold else 1
        return classification, d_score