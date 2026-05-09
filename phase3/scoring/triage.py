"""
Triage Mapping — SmileGuard Phase 3.2

Maps numerical scores to actionable triage categories:
    - TMJ Risk: Low / Moderate / High
    - Caries Triage: Routine / Soon / Urgent

Output sudah include Bahasa Indonesia label & color code untuk app UI.

Author: Riexu (riexu)
Date: 2026-05-06
"""

from typing import Dict, Any


# === TMJ Risk thresholds (dari scoring_outline.md) ===

TMJ_THRESHOLDS = {
    'low': (0, 30),
    'moderate': (31, 60),
    'high': (61, 100),
}


def get_tmj_triage(risk_score: float) -> Dict[str, Any]:
    """
    Map TMJ risk score [0-100] ke triage category.
    
    Args:
        risk_score: float [0-100]
    
    Returns:
        dict: {
            'category': 'LOW' | 'MODERATE' | 'HIGH',
            'category_id': str (Bahasa Indonesia),
            'action_id': str (rekomendasi tindakan),
            'urgency_id': str (urgency level),
            'color': str (color code untuk UI),
            'score': float (original score)
        }
    
    Raises:
        ValueError: jika score diluar range [0-100]
    """
    if not (0 <= risk_score <= 100):
        raise ValueError(f"Risk score must be 0-100, got {risk_score}")
    
    if risk_score < 31:
        return {
            'category': 'LOW',
            'category_id': 'Risiko Rendah',
            'action_id': 'Periksa rutin (6 bulan sekali)',
            'urgency_id': 'Tidak mendesak',
            'color': 'green',
            'score': risk_score,
        }
    elif risk_score < 61:
        return {
            'category': 'MODERATE',
            'category_id': 'Risiko Sedang',
            'action_id': 'Konsultasi dalam 1-2 minggu',
            'urgency_id': 'Disarankan segera',
            'color': 'yellow',
            'score': risk_score,
        }
    else:
        return {
            'category': 'HIGH',
            'category_id': 'Risiko Tinggi',
            'action_id': 'Konsultasi minggu ini',
            'urgency_id': 'Perlu perhatian segera',
            'color': 'red',
            'score': risk_score,
        }


def get_caries_triage(image_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map caries detections ke triage category.
    
    Logic:
        - Any severe caries (D5/D6) → URGENT
        - Multiple moderate caries (D3/D4 ≥ 2) → SOON
        - Multiple early caries (D1/D2 ≥ 3) → SOON (monitoring)
        - Otherwise → ROUTINE
    
    Args:
        image_features: dict dari extract_image_features()
    
    Returns:
        dict: triage info
    """
    n_severe = image_features.get('n_severe_caries', 0)
    n_d3 = image_features.get('n_D3_enamel_breakdown', 0)
    n_d4 = image_features.get('n_D4_dentin_shadow', 0)
    n_moderate = n_d3 + n_d4
    n_d1 = image_features.get('n_D1_first_change', 0)
    n_d2 = image_features.get('n_D2_distinct_opacity', 0)
    n_early = n_d1 + n_d2
    
    if n_severe >= 1:
        return {
            'category': 'URGENT',
            'category_id': 'Mendesak',
            'action_id': 'Segera periksa ke dokter gigi minggu ini',
            'reasoning_id': f'Terdeteksi {n_severe} indikasi karies parah',
            'color': 'red',
            'n_severe': n_severe,
            'n_moderate': n_moderate,
            'n_early': n_early,
        }
    elif n_moderate >= 2:
        return {
            'category': 'SOON',
            'category_id': 'Disarankan Segera',
            'action_id': 'Periksa dalam 1-2 minggu',
            'reasoning_id': f'Terdeteksi {n_moderate} indikasi karies sedang',
            'color': 'orange',
            'n_severe': n_severe,
            'n_moderate': n_moderate,
            'n_early': n_early,
        }
    elif n_early >= 3:
        return {
            'category': 'SOON',
            'category_id': 'Pemantauan Dini',
            'action_id': 'Periksa dalam 1 bulan untuk monitoring',
            'reasoning_id': f'Terdeteksi {n_early} indikasi karies awal',
            'color': 'yellow',
            'n_severe': n_severe,
            'n_moderate': n_moderate,
            'n_early': n_early,
        }
    else:
        return {
            'category': 'ROUTINE',
            'category_id': 'Pemeriksaan Rutin',
            'action_id': 'Cukup periksa rutin 6 bulan sekali',
            'reasoning_id': 'Tidak terdeteksi indikasi karies signifikan',
            'color': 'green',
            'n_severe': n_severe,
            'n_moderate': n_moderate,
            'n_early': n_early,
        }
