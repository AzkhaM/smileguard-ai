"""
Questionnaire Scorer — SmileGuard Phase 3.2

Computes 6 sub-scores dari kuesioner (18 items adaptive).
Reference: scoring_outline.md, DC/TMD literature.

Functions:
    - calculate_tmj_pain_subscore(answers) -> float [0-100]
    - calculate_tmj_dysfunction_subscore(answers) -> float [0-100]
    - calculate_bruxism_subscore(answers) -> float [0-100]
    - calculate_stress_modifier(answers) -> float [1.0-1.4]
    - calculate_trauma_score(answers) -> float [0-15]
    - calculate_demographics_modifier(answers) -> float [0.7-1.8]

Author: Riexu (riexu)
Date: 2026-05-06
"""

from typing import Dict, Any


# === Constants from scoring_outline.md ===

ITEM_WEIGHTS = {
    'A1': 3.5,  # TMJ pain (primary)
    'A2': 3.0,  # Pain modified by movement
    'A3': 2.5,  # Headache in temple
    'A4': 3.5,  # Stiffness/locking (mechanical dysfunction)
    'A5': 2.5,  # Joint sound (clicking)
    'A6': 3.0,  # Morning stiffness
    'B1': 3.0,  # Sleep bruxism
    'B2': 2.5,  # Awake bruxism
}

# Trauma severity weights (more recent = higher)
TRAUMA_WEIGHTS = {0: 0, 1: 1, 2: 3, 3: 8, 4: 15}

AGE_MODIFIERS = {
    0: 0.7,  # < 18 (juvenile)
    1: 1.0,  # 18-30 (peak onset)
    2: 1.2,  # 31-45 (peak prevalence)
    3: 1.0,  # 46-60
    4: 0.8,  # > 60 (different etiology)
}

GENDER_MODIFIERS = {
    0: 1.0,   # Male
    1: 1.5,   # Female (2-3x prevalence)
    2: 1.0,   # Not specified
}


# === Sub-score functions ===

def calculate_tmj_pain_subscore(answers: Dict[str, Any]) -> float:
    """
    Calculate TMJ pain sub-score from items A1, A2, A3.
    
    Args:
        answers: dict dengan key 'A1', 'A2', 'A3', value 0-4 (Likert)
    
    Returns:
        float [0-100]: TMJ pain severity normalized
    
    Raises:
        KeyError: jika required items A1, A2, A3 tidak ada
        ValueError: jika value diluar range 0-4
    """
    required = ['A1', 'A2', 'A3']
    for item in required:
        if item not in answers:
            raise KeyError(f"Required item '{item}' missing in answers")
        if not (0 <= answers[item] <= 4):
            raise ValueError(f"Item '{item}' value must be 0-4, got {answers[item]}")
    
    weighted_sum = (
        answers['A1'] * ITEM_WEIGHTS['A1']
        + answers['A2'] * ITEM_WEIGHTS['A2']
        + answers['A3'] * ITEM_WEIGHTS['A3']
    )
    
    # Max possible: (4 * 3.5) + (4 * 3.0) + (4 * 2.5) = 36
    max_possible = 4 * (ITEM_WEIGHTS['A1'] + ITEM_WEIGHTS['A2'] + ITEM_WEIGHTS['A3'])
    
    return (weighted_sum / max_possible) * 100


def calculate_tmj_dysfunction_subscore(answers: Dict[str, Any]) -> float:
    """
    Calculate TMJ dysfunction sub-score from items A4, A5, A6.
    
    Args:
        answers: dict dengan A4, A5, A6
    
    Returns:
        float [0-100]: dysfunction severity
    """
    required = ['A4', 'A5', 'A6']
    for item in required:
        if item not in answers:
            raise KeyError(f"Required item '{item}' missing")
        if not (0 <= answers[item] <= 4):
            raise ValueError(f"Item '{item}' must be 0-4")
    
    weighted_sum = (
        answers['A4'] * ITEM_WEIGHTS['A4']
        + answers['A5'] * ITEM_WEIGHTS['A5']
        + answers['A6'] * ITEM_WEIGHTS['A6']
    )
    
    max_possible = 4 * (ITEM_WEIGHTS['A4'] + ITEM_WEIGHTS['A5'] + ITEM_WEIGHTS['A6'])
    return (weighted_sum / max_possible) * 100


def calculate_bruxism_subscore(answers: Dict[str, Any]) -> float:
    """
    Calculate bruxism sub-score from items B1, B2.
    
    Args:
        answers: dict dengan B1, B2
    
    Returns:
        float [0-100]: bruxism severity
    """
    required = ['B1', 'B2']
    for item in required:
        if item not in answers:
            raise KeyError(f"Required item '{item}' missing")
        if not (0 <= answers[item] <= 4):
            raise ValueError(f"Item '{item}' must be 0-4")
    
    weighted_sum = (
        answers['B1'] * ITEM_WEIGHTS['B1']
        + answers['B2'] * ITEM_WEIGHTS['B2']
    )
    
    max_possible = 4 * (ITEM_WEIGHTS['B1'] + ITEM_WEIGHTS['B2'])
    return (weighted_sum / max_possible) * 100


def calculate_stress_modifier(answers: Dict[str, Any]) -> float:
    """
    Calculate stress modifier from conditional items C1, C2, C3.
    
    Stress amplifies TMD severity (clinical literature).
    Hanya applied jika Section C answered (conditional triggered).
    
    Args:
        answers: dict, mungkin tidak punya C1-C3 (kalau conditional tidak triggered)
    
    Returns:
        float [1.0-1.4]: multiplier untuk base score
            - 1.0 = no modifier (Section C not answered)
            - 1.4 = max stress amplification
    """
    # Jika C1 tidak ada → conditional tidak triggered → no modifier
    if 'C1' not in answers:
        return 1.0
    
    # Get with default 0 (untuk handle partial answers)
    c1 = answers.get('C1', 0)
    c2 = answers.get('C2', 0)
    c3 = answers.get('C3', 0)
    
    # Validate range
    for label, val in [('C1', c1), ('C2', c2), ('C3', c3)]:
        if not (0 <= val <= 4):
            raise ValueError(f"Item '{label}' must be 0-4, got {val}")
    
    stress_total = c1 + c2 + c3  # Max: 12
    
    # Linear scaling: 0 → 1.0, 12 → 1.4
    modifier = 1.0 + (stress_total / 12) * 0.4
    return modifier


def calculate_trauma_score(answers: Dict[str, Any]) -> float:
    """
    Calculate trauma history score from item D1.
    
    More recent trauma = higher score (exponential weighting).
    
    Args:
        answers: dict, mungkin tidak punya D1
    
    Returns:
        float [0-15]: trauma severity score (additive to risk)
    """
    if 'D1' not in answers:
        return 0.0
    
    d1 = answers['D1']
    
    if d1 not in TRAUMA_WEIGHTS:
        raise ValueError(f"Item D1 must be 0-4, got {d1}")
    
    return float(TRAUMA_WEIGHTS[d1])


def calculate_demographics_modifier(answers: Dict[str, Any]) -> float:
    """
    Calculate demographics modifier from items E1 (age) and E3 (gender).
    
    Args:
        answers: dict dengan E1, E3
    
    Returns:
        float [0.7-1.8]: multiplier reflecting demographic risk
    """
    required = ['E1', 'E3']
    for item in required:
        if item not in answers:
            raise KeyError(f"Required item '{item}' missing")
    
    age = answers['E1']
    gender = answers['E3']
    
    if age not in AGE_MODIFIERS:
        raise ValueError(f"Item E1 must be 0-4, got {age}")
    if gender not in GENDER_MODIFIERS:
        raise ValueError(f"Item E3 must be 0-2, got {gender}")
    
    return AGE_MODIFIERS[age] * GENDER_MODIFIERS[gender]
