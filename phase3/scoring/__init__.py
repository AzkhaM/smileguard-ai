"""
SmileGuard Scoring Module

Public API untuk TMJ + Caries risk assessment.

Quick start:
    >>> from scoring import assess_patient
    >>> result = assess_patient(questionnaire_answers, yolo_detections)
    >>> print(result['tmj_risk']['category'])
    'MODERATE'

Modules:
    - questionnaire_scorer: Sub-score calculations
    - image_feature_extractor: Convert YOLO output → features
    - triage: Map scores → triage categories
    - risk_calculator: Master orchestrator
"""

from .risk_calculator import (
    calculate_tmj_risk_score,
    assess_patient,
)
from .questionnaire_scorer import (
    calculate_tmj_pain_subscore,
    calculate_tmj_dysfunction_subscore,
    calculate_bruxism_subscore,
    calculate_stress_modifier,
    calculate_trauma_score,
    calculate_demographics_modifier,
)
from .image_feature_extractor import (
    extract_image_features,
    count_detections_by_class,
    calculate_lr_asymmetry,
    calculate_wear_pattern_score,
)
from .triage import (
    get_tmj_triage,
    get_caries_triage,
)

__version__ = '0.1.0'
__author__ = 'Riexu (riexu)'

__all__ = [
    # Master API
    'calculate_tmj_risk_score',
    'assess_patient',
    # Sub-scores
    'calculate_tmj_pain_subscore',
    'calculate_tmj_dysfunction_subscore',
    'calculate_bruxism_subscore',
    'calculate_stress_modifier',
    'calculate_trauma_score',
    'calculate_demographics_modifier',
    # Image features
    'extract_image_features',
    'count_detections_by_class',
    'calculate_lr_asymmetry',
    'calculate_wear_pattern_score',
    # Triage
    'get_tmj_triage',
    'get_caries_triage',
]
