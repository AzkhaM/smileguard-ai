"""
Risk Calculator — SmileGuard Phase 3.2 (v0.2)

Master orchestrator: combines questionnaire sub-scores + image features
ke final TMJ Risk Score [0-100].

This is the main API function yang akan di-call oleh app.

Changelog:
    v0.2 (2026-05-06): Switch from multiplicative to additive modifiers
                      to fix score explosion (Ibu Linda issue).
                      Stress and demographics now contribute as bonuses,
                      not multipliers.

Author: Riexu (riexu)
Date: 2026-05-06
"""

from typing import Dict, Any, List, Optional
from .questionnaire_scorer import (
    calculate_tmj_pain_subscore,
    calculate_tmj_dysfunction_subscore,
    calculate_bruxism_subscore,
    calculate_stress_modifier,
    calculate_trauma_score,
    calculate_demographics_modifier,
)
from .image_feature_extractor import extract_image_features
from .triage import get_tmj_triage, get_caries_triage


# === Component weights untuk base score ===
# Note: Ini hand-tuned weights v0.2. Phase 3.4 akan replace dengan XGBoost.

COMPONENT_WEIGHTS = {
    'pain': 0.30,           # 30% — TMJ pain primary indicator
    'dysfunction': 0.30,    # 30% — Functional limitation
    'bruxism': 0.20,        # 20% — Strong correlate of TMD
    'image': 0.10,          # 10% — Image-derived (wear + asymmetry)
    'trauma': 0.10,         # 10% — Historical trauma
}

# === Modifier bonus ranges (additive, NOT multiplicative) ===
# Why additive? Multiplicative compound bisa "explode" base score.
# Example: base=43, stress_mod=1.27, demo_mod=1.8 → 43*2.28 = 98 (FALSE HIGH)
# Additive: base=43, stress_bonus=+6, demo_bonus=+10 → 59 (correct MODERATE)

MAX_STRESS_BONUS = 10.0    # Max +10 points dari stress (C1-C3 all 4)
MAX_DEMO_BONUS = 10.0      # Max +10 points dari demographics (peak risk)
MIN_DEMO_BONUS = -5.0      # Min -5 dari demographics (low-risk demographic)


def _stress_modifier_to_bonus(stress_mod: float) -> float:
    """
    Convert legacy stress modifier (1.0-1.4x) to additive bonus (0-10).
    
    Args:
        stress_mod: float [1.0, 1.4] from calculate_stress_modifier()
    
    Returns:
        float [0, 10]: additive bonus untuk final score
    """
    # Linear mapping: 1.0 → 0, 1.4 → 10
    # (stress_mod - 1.0) / 0.4 * 10
    bonus = max(0, (stress_mod - 1.0) / 0.4 * MAX_STRESS_BONUS)
    return min(MAX_STRESS_BONUS, bonus)


def _demographics_modifier_to_bonus(demo_mod: float) -> float:
    """
    Convert legacy demographics modifier (0.7-1.8x) to additive bonus (-5 to +10).
    
    Args:
        demo_mod: float [0.7, 1.8] from calculate_demographics_modifier()
    
    Returns:
        float [-5, +10]: additive bonus
            - Low-risk demographic (e.g. juvenile male, mod=0.7) → -5
            - Average (mod=1.0)                                  → 0
            - Peak risk (peak-age female, mod=1.8)               → +10
    """
    # Center at mod=1.0 → bonus=0
    if demo_mod >= 1.0:
        # Map [1.0, 1.8] → [0, 10]
        bonus = (demo_mod - 1.0) / 0.8 * MAX_DEMO_BONUS
        return min(MAX_DEMO_BONUS, bonus)
    else:
        # Map [0.7, 1.0] → [-5, 0]
        bonus = (demo_mod - 1.0) / 0.3 * abs(MIN_DEMO_BONUS)
        return max(MIN_DEMO_BONUS, bonus)


def calculate_tmj_risk_score(
    questionnaire_answers: Dict[str, Any],
    image_features: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Master scoring function (v0.2 - additive modifiers).
    
    Combines weighted sub-scores + additive modifier bonuses 
    ke final TMJ Risk Score.
    
    Formula:
        base = weighted_sum(pain, dysfunction, bruxism, image, trauma)
        final = base + stress_bonus + demo_bonus
        
        Where:
            stress_bonus: 0 to +10
            demo_bonus: -5 to +10
        
        Theoretical max: ~100 (base 80 + 10 + 10)
        Theoretical min: 0 (clamped)
    
    Args:
        questionnaire_answers: dict dengan keys A1-A6, B1-B2, [C1-C3, D1, D2,]
                              E1, E3 (E2 optional)
        image_features: optional dict dari extract_image_features().
                       Kalau None, image contribution = 0.
    
    Returns:
        float [0-100]: TMJ Risk Score
    
    Example:
        >>> answers = {
        ...     'A1': 2, 'A2': 1, 'A3': 0,
        ...     'A4': 1, 'A5': 2, 'A6': 1,
        ...     'B1': 1, 'B2': 0,
        ...     'E1': 2, 'E3': 1,
        ... }
        >>> score = calculate_tmj_risk_score(answers)
        >>> 0 <= score <= 100
        True
    """
    # === Sub-scores from questionnaire (always required) ===
    pain_score = calculate_tmj_pain_subscore(questionnaire_answers)
    dysfunction_score = calculate_tmj_dysfunction_subscore(questionnaire_answers)
    bruxism_score = calculate_bruxism_subscore(questionnaire_answers)
    
    # === Modifiers ===
    stress_mod = calculate_stress_modifier(questionnaire_answers)
    trauma_score = calculate_trauma_score(questionnaire_answers)
    demo_mod = calculate_demographics_modifier(questionnaire_answers)
    
    # === Image contribution (optional) ===
    if image_features is not None:
        wear = image_features.get('wear_pattern_score', 0)
        asymmetry = image_features.get('asymmetry_score', 0)
        image_contrib = min(100, (wear * 5) + (asymmetry * 50))
    else:
        image_contrib = 0
    
    # === Weighted base score ===
    base_score = (
        pain_score * COMPONENT_WEIGHTS['pain']
        + dysfunction_score * COMPONENT_WEIGHTS['dysfunction']
        + bruxism_score * COMPONENT_WEIGHTS['bruxism']
        + image_contrib * COMPONENT_WEIGHTS['image']
        + (trauma_score / 15 * 100) * COMPONENT_WEIGHTS['trauma']
    )
    
    # === Apply ADDITIVE modifiers (v0.2 fix) ===
    stress_bonus = _stress_modifier_to_bonus(stress_mod)
    demo_bonus = _demographics_modifier_to_bonus(demo_mod)
    
    final_score = base_score + stress_bonus + demo_bonus
    
    # === Clamp to [0, 100] ===
    return max(0.0, min(100.0, final_score))


def assess_patient(
    questionnaire_answers: Dict[str, Any],
    yolo_detections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    End-to-end patient assessment.
    
    Combines:
        1. Questionnaire scoring → TMJ risk
        2. Image features extraction → caries triage
        3. Combined output for app UI
    
    Args:
        questionnaire_answers: dict patient answers
        yolo_detections: list YOLO detections (optional)
    
    Returns:
        dict: comprehensive assessment dengan:
            - tmj_risk: dict (score + triage)
            - caries: dict (counts + triage)
            - sub_scores: dict (untuk transparency/explainability)
            - image_features: dict (kalau ada)
    """
    # Extract image features kalau ada
    if yolo_detections is not None:
        image_features = extract_image_features(yolo_detections)
    else:
        image_features = None
    
    # Calculate TMJ risk
    tmj_score = calculate_tmj_risk_score(questionnaire_answers, image_features)
    tmj_triage = get_tmj_triage(tmj_score)
    
    # Caries triage (dari image only)
    if image_features is not None:
        caries_triage = get_caries_triage(image_features)
    else:
        caries_triage = {
            'category': 'UNKNOWN',
            'category_id': 'Tidak ada foto',
            'action_id': 'Upload foto gigi untuk analisis karies',
            'color': 'gray',
        }
    
    # Sub-scores untuk transparency (penting untuk medical context)
    sub_scores = {
        'pain': calculate_tmj_pain_subscore(questionnaire_answers),
        'dysfunction': calculate_tmj_dysfunction_subscore(questionnaire_answers),
        'bruxism': calculate_bruxism_subscore(questionnaire_answers),
        'stress_modifier': calculate_stress_modifier(questionnaire_answers),
        'trauma_score': calculate_trauma_score(questionnaire_answers),
        'demographics_modifier': calculate_demographics_modifier(questionnaire_answers),
    }
    
    return {
        'tmj_risk': tmj_triage,
        'caries': caries_triage,
        'sub_scores': sub_scores,
        'image_features': image_features,
        'disclaimer': (
            'Hasil ini merupakan indikasi awal berdasarkan kuesioner dan '
            'analisis foto, BUKAN diagnosis medis. Untuk diagnosis pasti '
            'dan rencana perawatan, silakan konsultasikan dengan dokter gigi.'
        ),
    }
