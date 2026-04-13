"""
Unit tests for the TopologicalEvaluator (Stage 3).

These tests are CPU-only and cover the two critical edge-case families:
  1. OCR / Text labels  – multiple distinct text strings printed on the SAME
     physical object (e.g. spice bottle).  SAM may segment the whole bottle
     for every anchor, producing near-identical masks.  The pipeline must
     correctly classify this as MULTIPLE by leveraging anchor separation.
  2. Dense Captioning   – sub-parts of a single macro-object (e.g. 'screen'
     and 'keyboard' inside 'laptop').  The NMS must absorb the sub-parts and
     classify the image as SINGLE.
"""

import sys
import os
import numpy as np
import pytest

# Ensure the package root is on the path when running via pytest directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stage_3_topology import TopologicalEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator():
    return TopologicalEvaluator(w1_ciou=0.4, w2_conflict=0.6, w3_anchor=0.5, threshold=0.045)


def _make_rect_mask(H, W, r0, r1, c0, c1):
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask


# ---------------------------------------------------------------------------
# Weight sanity
# ---------------------------------------------------------------------------

def test_weights_sum_to_one(evaluator):
    assert abs(evaluator.W1 + evaluator.W2 + evaluator.W3 - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# OCR / Text-label edge cases
# ---------------------------------------------------------------------------

def test_ocr_same_mask_distant_anchors_is_multiple(evaluator):
    """
    Classic OCR failure mode:
    Both text-label candidates produce the *same* full-bottle SAM mask.
    The anchor points, however, point to the upper and lower text regions
    respectively.  The anchor-aware NMS must preserve both candidates, and
    the anchor-separation term in D_Score must push it above the threshold.
    """
    H, W = 480, 640
    bottle_mask = _make_rect_mask(H, W, 100, 380, 220, 420)

    anchors = [(320, 150), (320, 330)]   # upper text vs lower text
    pred, score = evaluator.evaluate(
        [bottle_mask.copy(), bottle_mask.copy()],
        anchor_points=anchors,
        image_size=(W, H),
        candidate_labels=["Marshak Creek", "Steak Rub"],
    )
    assert pred == 0, f"OCR case must be MULTIPLE; got Single (D_Score={score:.4f})"
    assert score > evaluator.threshold, (
        f"D_Score {score:.4f} must exceed threshold {evaluator.threshold}"
    )


def test_ocr_same_mask_dissimilar_labels_without_anchors_is_multiple(evaluator):
    """
    True-multiple OCR case must still be MULTIPLE even without anchor inputs:
    identical geometry + lexically distinct labels should survive dedup and be
    separated by semantic divergence in D_Score.
    """
    H, W = 480, 640
    bottle_mask = _make_rect_mask(H, W, 100, 380, 220, 420)
    pred, score = evaluator.evaluate(
        [bottle_mask.copy(), bottle_mask.copy()],
        candidate_labels=["Marshak Creek", "Steak Rub"],
    )
    assert pred == 0, f"OCR dissimilar labels must be MULTIPLE; got Single (D_Score={score:.4f})"
    assert score > evaluator.threshold


def test_ocr_same_mask_close_anchors_is_single(evaluator):
    """
    If both text candidates *and* their anchors are essentially collocated
    (degenerate OCR, same label detected twice), the result should be Single.
    """
    H, W = 480, 640
    bottle_mask = _make_rect_mask(H, W, 100, 380, 220, 420)

    anchors = [(320, 240), (322, 242)]   # virtually same location
    pred, score = evaluator.evaluate(
        [bottle_mask.copy(), bottle_mask.copy()],
        anchor_points=anchors,
        image_size=(W, H),
        candidate_labels=["Marshak Creek", "Marshak Creek"],
    )
    assert pred == 1, f"Collocated OCR must be SINGLE; got Multiple (D_Score={score:.4f})"


def test_duplicate_text_same_mask_far_anchors_is_multiple(evaluator):
    """
    Duplicate label text printed in distant regions should not be absorbed.
    """
    H, W = 480, 640
    bottle_mask = _make_rect_mask(H, W, 100, 380, 220, 420)
    sep = evaluator._pair_anchor_distance((320, 140), (320, 340), (W, H))
    assert sep > evaluator.DUP_TEXT_ANCHOR_SEP_THRESHOLD
    pred, score = evaluator.evaluate(
        [bottle_mask.copy(), bottle_mask.copy()],
        anchor_points=[(320, 140), (320, 340)],
        image_size=(W, H),
        candidate_labels=["cinnamon", "cinnamon"],
    )
    assert pred == 0, f"Distant duplicate text must be MULTIPLE; got Single (D_Score={score:.4f})"
    assert score > evaluator.threshold


def test_synonym_same_mask_distant_anchors_is_single(evaluator):
    """
    Synonymous labels for one macro-object should be absorbed even if
    peak-suppression pushed anchors apart.
    """
    H, W = 480, 640
    phone_mask = _make_rect_mask(H, W, 120, 370, 240, 430)
    pred, score = evaluator.evaluate(
        [phone_mask.copy(), phone_mask.copy()],
        anchor_points=[(280, 160), (420, 330)],
        image_size=(W, H),
        candidate_labels=["Nokia phone", "Nokia mobile phone"],
    )
    assert pred == 1, f"Synonym case must be SINGLE; got Multiple (D_Score={score:.4f})"


def test_synonym_same_mask_without_anchors_is_single(evaluator):
    """
    Synonym-like labels for one object should still collapse to SINGLE even
    when anchor metadata is unavailable.
    """
    H, W = 480, 640
    phone_mask = _make_rect_mask(H, W, 120, 370, 240, 430)
    pred, score = evaluator.evaluate(
        [phone_mask.copy(), phone_mask.copy()],
        candidate_labels=["Nokia phone", "mobile phone"],
    )
    assert pred == 1, f"Synonym-no-anchor case must be SINGLE; got Multiple (D_Score={score:.4f})"


# ---------------------------------------------------------------------------
# Dense-Captioning / sub-part edge cases
# ---------------------------------------------------------------------------

def test_laptop_screen_inside_parent_is_single(evaluator):
    """
    'screen' mask is fully contained within 'laptop' mask.
    Anchor for 'screen' is spatially close to the laptop centre.
    The NMS must absorb 'screen', leaving a single survivor → SINGLE.
    """
    H, W = 480, 640
    laptop_mask = _make_rect_mask(H, W, 50, 430, 50, 590)
    screen_mask = _make_rect_mask(H, W, 60, 270, 55, 585)   # inside laptop

    anchor_laptop = (320, 240)
    anchor_screen = (320, 165)   # close to laptop centre; sep < 0.15
    pred, score = evaluator.evaluate(
        [laptop_mask, screen_mask],
        anchor_points=[anchor_laptop, anchor_screen],
        image_size=(W, H),
    )
    assert pred == 1, f"Laptop+screen must be SINGLE; got Multiple (D_Score={score:.4f})"


def test_part_to_whole_distant_containment_is_single(evaluator):
    """
    Part-to-whole should be absorbed when containment is near-complete and
    child area is much smaller, even with distant anchors.
    """
    H, W = 600, 800
    couch_mask = _make_rect_mask(H, W, 100, 560, 80, 760)
    pillow_mask = _make_rect_mask(H, W, 330, 520, 520, 740)  # contained in couch's right region

    pred, score = evaluator.evaluate(
        [couch_mask, pillow_mask],
        anchor_points=[(180, 210), (700, 450)],  # sep > 0.15 image diagonal
        image_size=(W, H),
        candidate_labels=["couch", "blue pillow"],
    )
    assert pred == 1, f"Couch+pillow must be SINGLE; got Multiple (D_Score={score:.4f})"


# ---------------------------------------------------------------------------
# Genuinely separate objects
# ---------------------------------------------------------------------------

def test_two_separate_objects_is_multiple(evaluator):
    """
    A car on the left and a traffic light on the top-right are two distinct
    objects with non-overlapping masks and distant anchors → MULTIPLE.
    """
    H, W = 480, 640
    car_mask   = _make_rect_mask(H, W, 200, 460,  20, 280)
    light_mask = _make_rect_mask(H, W,  30, 200, 500, 600)

    pred, score = evaluator.evaluate(
        [car_mask, light_mask],
        anchor_points=[(150, 330), (550, 115)],
        image_size=(W, H),
    )
    assert pred == 0, f"Two separate objects must be MULTIPLE; got Single (D_Score={score:.4f})"


# ---------------------------------------------------------------------------
# Backward-compatibility (no anchor_points / image_size provided)
# ---------------------------------------------------------------------------

def test_backward_compat_no_anchors(evaluator):
    """
    Calling evaluate() without anchor_points must still work correctly via
    the pixel-only path (existing behaviour preserved).
    """
    H, W = 480, 640
    car_mask   = _make_rect_mask(H, W, 200, 460,  20, 280)
    light_mask = _make_rect_mask(H, W,  30, 200, 500, 600)

    pred, score = evaluator.evaluate([car_mask, light_mask])
    assert pred == 0, f"Backward-compat: two separate masks should be MULTIPLE (D_Score={score:.4f})"


def test_single_mask_returns_single(evaluator):
    H, W = 480, 640
    mask = _make_rect_mask(H, W, 100, 400, 100, 540)
    pred, score = evaluator.evaluate([mask])
    assert pred == 1
    assert score == 0.0


def test_all_empty_masks_returns_single(evaluator):
    H, W = 480, 640
    empty = np.zeros((H, W), dtype=np.uint8)
    pred, score = evaluator.evaluate([empty, empty])
    assert pred == 1
    assert score == 0.0


# ---------------------------------------------------------------------------
# _calculate_anchor_separation helper
# ---------------------------------------------------------------------------

def test_anchor_separation_normalized(evaluator):
    # Anchors at opposite corners of a 640×480 image
    image_size = (640, 480)
    sep = evaluator._calculate_anchor_separation([(0, 0), (640, 480)], image_size)
    assert abs(sep - 1.0) < 1e-6, f"Corner-to-corner sep should be 1.0, got {sep}"


def test_anchor_separation_same_point(evaluator):
    sep = evaluator._calculate_anchor_separation([(320, 240), (320, 240)], (640, 480))
    assert sep == 0.0


def test_anchor_separation_single_point(evaluator):
    sep = evaluator._calculate_anchor_separation([(100, 100)], (640, 480))
    assert sep == 0.0
