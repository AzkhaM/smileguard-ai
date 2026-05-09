"""
Test Suite — SmileGuard Scoring Module

Run dengan: pytest tests/test_scoring.py -v

Test coverage:
    - Boundary cases (all 0s, all 4s, missing items)
    - Sanity checks (sehat → low score, severe → high score)
    - Validation errors (out of range, missing required)
    - Image feature extraction
    - End-to-end assess_patient

Author: Riexu (riexu)
Date: 2026-05-06
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from scoring import (
    calculate_tmj_pain_subscore,
    calculate_tmj_dysfunction_subscore,
    calculate_bruxism_subscore,
    calculate_stress_modifier,
    calculate_trauma_score,
    calculate_demographics_modifier,
    calculate_tmj_risk_score,
    extract_image_features,
    count_detections_by_class,
    calculate_lr_asymmetry,
    calculate_wear_pattern_score,
    get_tmj_triage,
    get_caries_triage,
    assess_patient,
)


# === FIXTURES (reusable test data) ===

@pytest.fixture
def healthy_patient_answers():
    """Pasien sehat: semua jawaban 0 (no symptoms)."""
    return {
        'A1': 0, 'A2': 0, 'A3': 0,
        'A4': 0, 'A5': 0, 'A6': 0,
        'B1': 0, 'B2': 0,
        'E1': 1,  # 18-30
        'E2': 0,  # < 6 bulan visit
        'E3': 0,  # Male
    }


@pytest.fixture
def severe_patient_answers():
    """Pasien dengan gejala parah: semua jawaban 4 + conditional + trauma."""
    return {
        'A1': 4, 'A2': 4, 'A3': 4,
        'A4': 4, 'A5': 4, 'A6': 4,
        'B1': 4, 'B2': 4,
        'C1': 4, 'C2': 4, 'C3': 4,
        'D1': 4, 'D2': 0,
        'E1': 2, 'E2': 4, 'E3': 1,  # 31-45, female (high prevalence)
    }


@pytest.fixture
def moderate_patient_answers():
    """Pasien moderate: gejala sedang."""
    return {
        'A1': 2, 'A2': 2, 'A3': 1,
        'A4': 2, 'A5': 2, 'A6': 2,
        'B1': 2, 'B2': 1,
        'C1': 2, 'C2': 1, 'C3': 1,
        'E1': 2, 'E2': 2, 'E3': 1,
    }


@pytest.fixture
def healthy_detections():
    """Image dengan no caries (semua D0 sound)."""
    return [
        {'class': 'D0_sound', 'confidence': 0.95, 'bbox': [0.1, 0.1, 0.2, 0.2]},
        {'class': 'D0_sound', 'confidence': 0.92, 'bbox': [0.3, 0.1, 0.4, 0.2]},
    ]


@pytest.fixture
def severe_caries_detections():
    """Image dengan severe caries (D5, D6)."""
    return [
        {'class': 'D5_distinct_cavity', 'confidence': 0.88, 'bbox': [0.15, 0.2, 0.25, 0.3]},
        {'class': 'D6_extensive_cavity', 'confidence': 0.91, 'bbox': [0.4, 0.3, 0.5, 0.4]},
        {'class': 'D3_enamel_breakdown', 'confidence': 0.75, 'bbox': [0.6, 0.2, 0.7, 0.3]},
    ]


@pytest.fixture
def asymmetric_detections():
    """Detections clustered di satu sisi (left side only)."""
    return [
        {'class': 'D3_enamel_breakdown', 'confidence': 0.8, 'bbox': [0.1, 0.1, 0.2, 0.2]},
        {'class': 'D4_dentin_shadow', 'confidence': 0.85, 'bbox': [0.15, 0.3, 0.25, 0.4]},
        {'class': 'D3_enamel_breakdown', 'confidence': 0.75, 'bbox': [0.05, 0.5, 0.15, 0.6]},
    ]


# ============================================================
# TEST: questionnaire_scorer.py
# ============================================================

class TestTMJPainSubscore:
    """Tests untuk calculate_tmj_pain_subscore."""
    
    def test_zero_pain(self, healthy_patient_answers):
        """Pasien tanpa gejala harusnya skor 0."""
        score = calculate_tmj_pain_subscore(healthy_patient_answers)
        assert score == 0.0
    
    def test_max_pain(self, severe_patient_answers):
        """Pasien dengan max severity harusnya skor 100."""
        score = calculate_tmj_pain_subscore(severe_patient_answers)
        assert score == 100.0
    
    def test_moderate_pain(self):
        """Pasien moderate: skor di tengah-tengah range."""
        answers = {'A1': 2, 'A2': 2, 'A3': 2}
        score = calculate_tmj_pain_subscore(answers)
        assert 40 < score < 60  # roughly half
    
    def test_missing_required(self):
        """Missing A1 harus raise KeyError."""
        with pytest.raises(KeyError):
            calculate_tmj_pain_subscore({'A2': 1, 'A3': 1})
    
    def test_invalid_value(self):
        """Value > 4 harus raise ValueError."""
        with pytest.raises(ValueError):
            calculate_tmj_pain_subscore({'A1': 5, 'A2': 0, 'A3': 0})
    
    def test_negative_value(self):
        """Negative value harus raise ValueError."""
        with pytest.raises(ValueError):
            calculate_tmj_pain_subscore({'A1': -1, 'A2': 0, 'A3': 0})
    
    def test_returns_float_in_range(self):
        """Output selalu float dalam range [0, 100]."""
        for a1 in range(5):
            for a2 in range(5):
                for a3 in range(5):
                    score = calculate_tmj_pain_subscore({'A1': a1, 'A2': a2, 'A3': a3})
                    assert 0 <= score <= 100
                    assert isinstance(score, float)


class TestTMJDysfunctionSubscore:
    """Tests untuk calculate_tmj_dysfunction_subscore."""
    
    def test_zero_dysfunction(self, healthy_patient_answers):
        score = calculate_tmj_dysfunction_subscore(healthy_patient_answers)
        assert score == 0.0
    
    def test_max_dysfunction(self, severe_patient_answers):
        score = calculate_tmj_dysfunction_subscore(severe_patient_answers)
        assert score == 100.0
    
    def test_missing_required(self):
        with pytest.raises(KeyError):
            calculate_tmj_dysfunction_subscore({'A4': 1, 'A5': 1})


class TestBruxismSubscore:
    """Tests untuk calculate_bruxism_subscore."""
    
    def test_zero_bruxism(self, healthy_patient_answers):
        score = calculate_bruxism_subscore(healthy_patient_answers)
        assert score == 0.0
    
    def test_max_bruxism(self, severe_patient_answers):
        score = calculate_bruxism_subscore(severe_patient_answers)
        assert score == 100.0


class TestStressModifier:
    """Tests untuk calculate_stress_modifier (conditional)."""
    
    def test_no_stress_section(self):
        """Kalau Section C tidak ada → modifier 1.0."""
        answers = {'A1': 0}
        assert calculate_stress_modifier(answers) == 1.0
    
    def test_zero_stress(self):
        """C1, C2, C3 = 0 → modifier 1.0."""
        answers = {'C1': 0, 'C2': 0, 'C3': 0}
        assert calculate_stress_modifier(answers) == 1.0
    
    def test_max_stress(self):
        """C1, C2, C3 = 4 → modifier 1.4."""
        answers = {'C1': 4, 'C2': 4, 'C3': 4}
        assert calculate_stress_modifier(answers) == pytest.approx(1.4, rel=1e-3)
    
    def test_modifier_range(self):
        """Modifier selalu di [1.0, 1.4]."""
        for c1 in range(5):
            for c2 in range(5):
                for c3 in range(5):
                    mod = calculate_stress_modifier({'C1': c1, 'C2': c2, 'C3': c3})
                    assert 1.0 <= mod <= 1.4


class TestTraumaScore:
    """Tests untuk calculate_trauma_score (conditional)."""
    
    def test_no_trauma_section(self):
        """Kalau D1 tidak ada → 0."""
        assert calculate_trauma_score({'A1': 0}) == 0.0
    
    def test_no_trauma(self):
        assert calculate_trauma_score({'D1': 0}) == 0.0
    
    def test_recent_trauma_max(self):
        assert calculate_trauma_score({'D1': 4}) == 15.0
    
    def test_old_trauma_low(self):
        """Old trauma > 5 years → score rendah."""
        assert calculate_trauma_score({'D1': 1}) == 1.0
    
    def test_invalid_d1(self):
        with pytest.raises(ValueError):
            calculate_trauma_score({'D1': 5})


class TestDemographicsModifier:
    """Tests untuk calculate_demographics_modifier."""
    
    def test_young_male(self):
        """18-30 male: 1.0 * 1.0 = 1.0."""
        assert calculate_demographics_modifier({'E1': 1, 'E3': 0}) == 1.0
    
    def test_peak_age_female(self):
        """31-45 female: 1.2 * 1.5 = 1.8 (max risk)."""
        assert calculate_demographics_modifier({'E1': 2, 'E3': 1}) == pytest.approx(1.8)
    
    def test_juvenile(self):
        """< 18: 0.7 modifier."""
        assert calculate_demographics_modifier({'E1': 0, 'E3': 0}) == 0.7
    
    def test_missing_required(self):
        with pytest.raises(KeyError):
            calculate_demographics_modifier({'E1': 1})


# ============================================================
# TEST: image_feature_extractor.py
# ============================================================

class TestCountDetections:
    """Tests untuk count_detections_by_class."""
    
    def test_empty_detections(self):
        counts = count_detections_by_class([])
        assert counts['n_D0_sound'] == 0
        assert counts['n_D6_extensive_cavity'] == 0
    
    def test_single_class(self, severe_caries_detections):
        counts = count_detections_by_class(severe_caries_detections)
        assert counts['n_D5_distinct_cavity'] == 1
        assert counts['n_D6_extensive_cavity'] == 1
        assert counts['n_D3_enamel_breakdown'] == 1
        assert counts['n_D0_sound'] == 0
    
    def test_unknown_class_ignored(self):
        """Class yang tidak ada di ICDAS list di-ignore."""
        detections = [
            {'class': 'unknown_class', 'confidence': 0.5, 'bbox': [0, 0, 1, 1]},
            {'class': 'D5_distinct_cavity', 'confidence': 0.8, 'bbox': [0, 0, 1, 1]},
        ]
        counts = count_detections_by_class(detections)
        assert counts['n_D5_distinct_cavity'] == 1


class TestAsymmetry:
    """Tests untuk calculate_lr_asymmetry."""
    
    def test_empty(self):
        assert calculate_lr_asymmetry([]) == 0.0
    
    def test_perfectly_asymmetric(self, asymmetric_detections):
        """Semua di kiri (x_center < 0.5) → asymmetry 1.0."""
        score = calculate_lr_asymmetry(asymmetric_detections)
        assert score == 1.0
    
    def test_perfectly_symmetric(self):
        """Sama jumlah kiri dan kanan → 0."""
        detections = [
            {'class': 'D3_enamel_breakdown', 'confidence': 0.8, 'bbox': [0.1, 0.1, 0.2, 0.2]},  # left
            {'class': 'D3_enamel_breakdown', 'confidence': 0.8, 'bbox': [0.7, 0.1, 0.8, 0.2]},  # right
        ]
        assert calculate_lr_asymmetry(detections) == 0.0
    
    def test_only_d0_no_asymmetry(self, healthy_detections):
        """D0 sound tidak dihitung untuk asymmetry."""
        assert calculate_lr_asymmetry(healthy_detections) == 0.0


class TestWearPattern:
    """Tests untuk calculate_wear_pattern_score."""
    
    def test_zero_wear(self):
        counts = {'n_D3_enamel_breakdown': 0, 'n_D4_dentin_shadow': 0}
        assert calculate_wear_pattern_score(counts) == 0
    
    def test_d4_weighted_higher(self):
        """D4 punya weight 1.5x."""
        counts1 = {'n_D3_enamel_breakdown': 1, 'n_D4_dentin_shadow': 0}
        counts2 = {'n_D3_enamel_breakdown': 0, 'n_D4_dentin_shadow': 1}
        assert calculate_wear_pattern_score(counts2) > calculate_wear_pattern_score(counts1)


class TestImageFeatureExtraction:
    """Tests untuk extract_image_features (master function)."""
    
    def test_empty_returns_zeros(self):
        features = extract_image_features([])
        assert features['n_caries_total'] == 0
        assert features['n_severe_caries'] == 0
        assert features['avg_confidence'] == 0
    
    def test_severe_caries_counted(self, severe_caries_detections):
        features = extract_image_features(severe_caries_detections)
        assert features['n_severe_caries'] == 2  # D5 + D6
        assert features['n_caries_total'] == 3
        assert features['avg_confidence'] > 0.7


# ============================================================
# TEST: triage.py
# ============================================================

class TestTMJTriage:
    """Tests untuk get_tmj_triage."""
    
    def test_low_risk(self):
        result = get_tmj_triage(15)
        assert result['category'] == 'LOW'
        assert result['color'] == 'green'
    
    def test_moderate_risk(self):
        result = get_tmj_triage(45)
        assert result['category'] == 'MODERATE'
        assert result['color'] == 'yellow'
    
    def test_high_risk(self):
        result = get_tmj_triage(85)
        assert result['category'] == 'HIGH'
        assert result['color'] == 'red'
    
    def test_boundary_low_moderate(self):
        """Score 30 = LOW, 31 = MODERATE."""
        assert get_tmj_triage(30)['category'] == 'LOW'
        assert get_tmj_triage(31)['category'] == 'MODERATE'
    
    def test_boundary_moderate_high(self):
        assert get_tmj_triage(60)['category'] == 'MODERATE'
        assert get_tmj_triage(61)['category'] == 'HIGH'
    
    def test_invalid_score(self):
        with pytest.raises(ValueError):
            get_tmj_triage(150)
        with pytest.raises(ValueError):
            get_tmj_triage(-10)


class TestCariesTriage:
    """Tests untuk get_caries_triage."""
    
    def test_no_caries_routine(self):
        features = {'n_severe_caries': 0, 'n_D3_enamel_breakdown': 0,
                    'n_D4_dentin_shadow': 0, 'n_D1_first_change': 0,
                    'n_D2_distinct_opacity': 0}
        result = get_caries_triage(features)
        assert result['category'] == 'ROUTINE'
    
    def test_severe_urgent(self):
        features = {'n_severe_caries': 1}
        result = get_caries_triage(features)
        assert result['category'] == 'URGENT'
        assert result['color'] == 'red'
    
    def test_multiple_moderate_soon(self):
        features = {'n_severe_caries': 0, 'n_D3_enamel_breakdown': 1,
                    'n_D4_dentin_shadow': 1, 'n_D1_first_change': 0,
                    'n_D2_distinct_opacity': 0}
        result = get_caries_triage(features)
        assert result['category'] == 'SOON'


# ============================================================
# TEST: risk_calculator.py (END-TO-END)
# ============================================================

class TestEndToEnd:
    """End-to-end tests untuk assess_patient."""
    
    def test_healthy_patient_low_risk(self, healthy_patient_answers, healthy_detections):
        """Pasien sehat + foto bagus → LOW risk + ROUTINE caries."""
        result = assess_patient(healthy_patient_answers, healthy_detections)
        assert result['tmj_risk']['category'] == 'LOW'
        assert result['caries']['category'] == 'ROUTINE'
        assert 'disclaimer' in result
    
    def test_severe_patient_high_risk(self, severe_patient_answers, severe_caries_detections):
        """Pasien parah + caries severe → HIGH + URGENT."""
        result = assess_patient(severe_patient_answers, severe_caries_detections)
        assert result['tmj_risk']['category'] == 'HIGH'
        assert result['caries']['category'] == 'URGENT'
    
    def test_no_image_returns_unknown_caries(self, healthy_patient_answers):
        """Tanpa foto → caries category = UNKNOWN."""
        result = assess_patient(healthy_patient_answers)
        assert result['caries']['category'] == 'UNKNOWN'
        assert result['image_features'] is None
    
    def test_disclaimer_always_present(self, healthy_patient_answers):
        """Disclaimer wajib ada di setiap response."""
        result = assess_patient(healthy_patient_answers)
        assert 'disclaimer' in result
        assert 'BUKAN diagnosis' in result['disclaimer']
    
    def test_score_in_valid_range(self, healthy_patient_answers, severe_patient_answers):
        """TMJ score selalu 0-100."""
        for answers in [healthy_patient_answers, severe_patient_answers]:
            score = calculate_tmj_risk_score(answers)
            assert 0 <= score <= 100
    
    def test_severe_score_higher_than_healthy(
        self, healthy_patient_answers, severe_patient_answers
    ):
        """Sanity: severe patient harus dapat score lebih tinggi dari healthy."""
        score_healthy = calculate_tmj_risk_score(healthy_patient_answers)
        score_severe = calculate_tmj_risk_score(severe_patient_answers)
        assert score_severe > score_healthy
    
    def test_sub_scores_in_output(self, moderate_patient_answers):
        """Output harus include sub_scores untuk transparency."""
        result = assess_patient(moderate_patient_answers)
        assert 'sub_scores' in result
        assert 'pain' in result['sub_scores']
        assert 'dysfunction' in result['sub_scores']
        assert 'bruxism' in result['sub_scores']


# ============================================================
# TEST: edge cases & robustness
# ============================================================

class TestEdgeCases:
    """Edge cases & robustness tests."""
    
    def test_partial_conditional_answers(self):
        """Pasien jawab core saja, conditional skip."""
        minimal_answers = {
            'A1': 1, 'A2': 0, 'A3': 0,
            'A4': 0, 'A5': 0, 'A6': 0,
            'B1': 0, 'B2': 0,
            'E1': 1, 'E3': 0,
        }
        # Tidak boleh raise error
        score = calculate_tmj_risk_score(minimal_answers)
        assert 0 <= score <= 100
    
    def test_full_conditional_answers(self, severe_patient_answers):
        """Semua items dijawab termasuk conditional."""
        score = calculate_tmj_risk_score(severe_patient_answers)
        assert score > 80  # severe patient harusnya >80


if __name__ == '__main__':
    # Quick smoke test kalau run tanpa pytest
    print('Running quick smoke test...')
    
    healthy = {
        'A1': 0, 'A2': 0, 'A3': 0,
        'A4': 0, 'A5': 0, 'A6': 0,
        'B1': 0, 'B2': 0,
        'E1': 1, 'E3': 0,
    }
    print(f'Healthy patient TMJ score: {calculate_tmj_risk_score(healthy):.2f}')
    
    severe = {
        'A1': 4, 'A2': 4, 'A3': 4,
        'A4': 4, 'A5': 4, 'A6': 4,
        'B1': 4, 'B2': 4,
        'C1': 4, 'C2': 4, 'C3': 4,
        'D1': 4,
        'E1': 2, 'E3': 1,
    }
    print(f'Severe patient TMJ score: {calculate_tmj_risk_score(severe):.2f}')
    
    print('Run "pytest tests/test_scoring.py -v" untuk full test suite.')
