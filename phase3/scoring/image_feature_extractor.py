"""
Image Feature Extractor — SmileGuard Phase 3.2

Converts YOLO Phase 1 model output → tabular features untuk fusion model.

Input format (per image):
    detections = [
        {'class': 'D5_distinct_cavity', 'confidence': 0.87, 'bbox': [x1, y1, x2, y2]},
        ...
    ]

Output: dict of features yang bisa digabung dengan questionnaire scores.

Author: Riexu (riexu)
Date: 2026-05-06
"""

from typing import List, Dict, Any


# === ICDAS class names (consistent dengan Phase 1 model) ===

ICDAS_CLASSES = [
    'D0_sound',
    'D1_first_change',
    'D2_distinct_opacity',
    'D3_enamel_breakdown',
    'D4_dentin_shadow',
    'D5_distinct_cavity',
    'D6_extensive_cavity',
]

# Severity classes (exclude D0 which is healthy)
SEVERITY_CLASSES = ICDAS_CLASSES[1:]  # D1-D6

# Severe caries (urgency indicators)
SEVERE_CLASSES = ['D5_distinct_cavity', 'D6_extensive_cavity']

# Wear pattern indicators
WEAR_PATTERN_CLASSES = ['D3_enamel_breakdown', 'D4_dentin_shadow']


def count_detections_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count detections per ICDAS class.
    
    Args:
        detections: list of detection dicts
    
    Returns:
        dict: {'n_D0_sound': int, 'n_D1_first_change': int, ...}
    """
    counts = {f'n_{cls}': 0 for cls in ICDAS_CLASSES}
    
    for det in detections:
        cls = det.get('class')
        if cls in ICDAS_CLASSES:
            counts[f'n_{cls}'] += 1
    
    return counts


def calculate_lr_asymmetry(detections: List[Dict[str, Any]]) -> float:
    """
    Calculate left-right asymmetry of caries distribution.
    
    Caries clustering on one side = potential asymmetric mastication
    (TMJ dysfunction indicator).
    
    Args:
        detections: list of detection dicts dengan bbox [x1, y1, x2, y2]
                    bbox coordinates assumed normalized [0-1] atau pixel.
                    Asumsi: x_center 0.5 = midline.
    
    Returns:
        float [0-1]:
            0 = perfectly symmetric (sama jumlah kiri-kanan)
            1 = all on one side
    """
    # Filter only severity detections (skip D0_sound)
    severity_dets = [
        d for d in detections 
        if d.get('class') in SEVERITY_CLASSES
    ]
    
    if not severity_dets:
        return 0.0
    
    left_count = 0
    right_count = 0
    
    for det in severity_dets:
        bbox = det.get('bbox')
        if bbox is None or len(bbox) < 4:
            continue
        
        # Calculate x_center
        x_center = (bbox[0] + bbox[2]) / 2
        
        # Determine side (assumes midline at 0.5 if normalized, or image_width/2 if pixel)
        # Heuristic: if bbox values < 1.0, assume normalized
        is_normalized = all(0 <= b <= 1 for b in bbox)
        midline = 0.5 if is_normalized else _estimate_midline(severity_dets)
        
        if x_center < midline:
            left_count += 1
        else:
            right_count += 1
    
    total = left_count + right_count
    if total == 0:
        return 0.0
    
    return abs(left_count - right_count) / total


def _estimate_midline(detections: List[Dict[str, Any]]) -> float:
    """Helper: estimate image midline dari max bbox x-coordinate."""
    max_x = max(
        max(d['bbox'][0], d['bbox'][2]) 
        for d in detections 
        if 'bbox' in d
    )
    return max_x / 2


def calculate_wear_pattern_score(class_counts: Dict[str, int]) -> float:
    """
    Calculate wear pattern indicator from D3-D4 distribution.
    
    Multiple D3 (enamel breakdown) + D4 (dentin shadow) across teeth =
    indication of bruxism wear pattern.
    
    Args:
        class_counts: output dari count_detections_by_class()
    
    Returns:
        float [0-N]: weighted count of wear-indicator caries
    """
    n_d3 = class_counts.get('n_D3_enamel_breakdown', 0)
    n_d4 = class_counts.get('n_D4_dentin_shadow', 0)
    
    # D4 weighted higher (more advanced wear)
    return n_d3 + n_d4 * 1.5


def extract_image_features(detections: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Master function: extract semua image-derived features.
    
    Args:
        detections: list of YOLO detection dicts
    
    Returns:
        dict: comprehensive image features untuk fusion model
    """
    # Per-class counts
    counts = count_detections_by_class(detections)
    
    # Aggregate features
    features = dict(counts)  # copy class counts
    
    # Total caries (exclude D0_sound)
    features['n_caries_total'] = sum(
        counts[f'n_{cls}'] for cls in SEVERITY_CLASSES
    )
    
    # Severe caries (urgency signal)
    features['n_severe_caries'] = sum(
        counts[f'n_{cls}'] for cls in SEVERE_CLASSES
    )
    
    # Wear pattern score
    features['wear_pattern_score'] = calculate_wear_pattern_score(counts)
    
    # Asymmetry score
    features['asymmetry_score'] = calculate_lr_asymmetry(detections)
    
    # Average confidence (kalau ada detection)
    if detections:
        confidences = [d.get('confidence', 0) for d in detections]
        features['avg_confidence'] = sum(confidences) / len(confidences)
    else:
        features['avg_confidence'] = 0.0
    
    return features
