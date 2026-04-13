import numpy as np
import cv2
import re
from difflib import SequenceMatcher

class TopologicalEvaluator:
    LABEL_STOPWORDS = {"a", "an", "the", "of", "on", "in", "for", "with", "and", "at", "by", "to", "from", "or", "but", "as", "if"}
    SEMANTIC_JACCARD_THRESHOLD = 0.55
    SEMANTIC_CONTAINMENT_THRESHOLD = 0.5
    SEMANTIC_CHAR_SIM_THRESHOLD = 0.78
    ABSORB_MIN_IOM = 0.85
    SAME_MASK_IOU = 0.98
    STRONG_CONTAINMENT_IOM = 0.98
    STRONG_CONTAINMENT_AREA_RATIO = 0.6

    def __init__(self, w1_ciou=0.4, w2_conflict=0.6, w3_anchor=0.5, w3_semantic=None, threshold=0.045):
        print("🚀 Initializing VizWiz Geometry+Semantic Topological Evaluator...")
        if w3_semantic is None:
            w3_semantic = w3_anchor  # Backward compatibility with existing config key
        total_weight = w1_ciou + w2_conflict + w3_semantic
        self.W1 = w1_ciou / total_weight
        self.W2 = w2_conflict / total_weight
        self.W3 = w3_semantic / total_weight
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

    def _calculate_pixel_iou(self, maskA, maskB):
        """Calculates pixel-perfect Intersection over Union."""
        mA = maskA > 0
        mB = maskB > 0
        intersection = np.logical_and(mA, mB).sum()
        union = np.logical_or(mA, mB).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def _normalize_label(self, text):
        if text is None:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def _tokenize_label(self, text):
        norm = self._normalize_label(text)
        if not norm:
            return set()
        return {tok for tok in norm.split() if tok and tok not in self.LABEL_STOPWORDS}

    def _are_semantically_equivalent(self, label_a, label_b):
        score = self._semantic_similarity(label_a, label_b)
        if score >= self.SEMANTIC_CHAR_SIM_THRESHOLD:
            return True

        ta = self._tokenize_label(label_a)
        tb = self._tokenize_label(label_b)
        if not ta or not tb:
            return False

        if ta == tb:
            return True

        inter = ta.intersection(tb)
        if not inter:
            return False

        jaccard = len(inter) / len(ta.union(tb))
        containment = min(len(inter) / len(ta), len(inter) / len(tb))
        # Lexical-equivalence rule for synonyms/aliases, using only lightweight string math.
        return jaccard >= self.SEMANTIC_JACCARD_THRESHOLD or containment >= self.SEMANTIC_CONTAINMENT_THRESHOLD

    def _semantic_similarity(self, label_a, label_b):
        a = self._normalize_label(label_a)
        b = self._normalize_label(label_b)
        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        ta = self._tokenize_label(a)
        tb = self._tokenize_label(b)
        if ta and tb:
            inter = len(ta.intersection(tb))
            union = len(ta.union(tb))
            jaccard = inter / union if union > 0 else 0.0
        else:
            jaccard = 0.0

        char_ratio = SequenceMatcher(None, a, b).ratio()
        return max(char_ratio, jaccard)

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

    def _calculate_semantic_divergence(self, labels):
        if labels is None or len(labels) < 2:
            return 0.0

        sims = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                sims.append(self._semantic_similarity(labels[i], labels[j]))

        if not sims:
            return 0.0
        avg_sim = sum(sims) / len(sims)
        return 1.0 - avg_sim

    def evaluate(self, masks, anchor_points=None, image_size=None, candidate_labels=None):
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
        valid_labels = (
            [candidate_labels[i] for i in valid_indices]
            if candidate_labels is not None and len(candidate_labels) == len(masks)
            else None
        )

        if len(valid_masks) < 2:
            return 1, 0.0

        # --- SOLID-PIXEL GEOMETRY+SEMANTIC NMS ---
        # Fill internal holes so that parent objects (e.g. couch) absorb their
        # children (e.g. pillow) correctly.
        # If two masks are almost identical, absorb only when labels are lexically
        # equivalent; otherwise keep both so OCR-like distinct strings survive.
        solid_masks = [self._fill_holes(m) for m in valid_masks]
        areas = [np.sum(m) for m in solid_masks]

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
                if iom < self.ABSORB_MIN_IOM:
                    continue

                iou = self._calculate_pixel_iou(mask_parent, mask_child)
                area_parent = areas[j]
                area_child = areas[i]
                area_ratio = min(area_parent, area_child) / max(area_parent, area_child)
                same_shape = iou >= self.SAME_MASK_IOU
                strong_containment = (
                    iom >= self.STRONG_CONTAINMENT_IOM and area_ratio <= self.STRONG_CONTAINMENT_AREA_RATIO
                )
                semantic_match = (
                    self._are_semantically_equivalent(valid_labels[i], valid_labels[j])
                    if valid_labels is not None
                    else True
                )

                # Part-to-whole should always absorb when containment is near-complete.
                if strong_containment:
                    is_absorbed = True
                    break

                # Near-identical geometry: absorb only when labels are equivalent.
                if same_shape:
                    if semantic_match:
                        is_absorbed = True
                        break
                    continue

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
        final_labels = (
            [valid_labels[idx] for idx in kept_indices]
            if valid_labels is not None
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
        semantic_divergence = self._calculate_semantic_divergence(final_labels)

        d_score = (
            (self.W1 * (1.0 - avg_overlap))
            + (self.W2 * conflict_ratio)
            + (self.W3 * semantic_divergence)
        )

        classification = 0 if d_score >= self.threshold else 1
        return classification, d_score
