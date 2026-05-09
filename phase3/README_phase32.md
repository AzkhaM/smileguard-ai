# Phase 3.2 — Scoring Module Implementation

**Date:** 2026-05-06  
**Status:** ✅ Complete  
**Test Coverage:** 54/54 passing (100%)

## 📦 Structure

```
phase3/
├── questionnaire_id.json           # Phase 3.1 deliverable
├── scoring_outline.md              # Phase 3.1 deliverable
├── README_phase32.md               # this file
├── 04_phase3_scoring_demo.ipynb    # demo notebook with 5 personas
│
├── scoring/                        # Library module (importable)
│   ├── __init__.py                 # public API
│   ├── questionnaire_scorer.py    # 6 sub-score functions
│   ├── image_feature_extractor.py # YOLO output → features
│   ├── triage.py                   # score → category mapping
│   └── risk_calculator.py          # master orchestrator
│
└── tests/
    └── test_scoring.py             # pytest suite (54 tests)
```

## 🚀 Quick Start

### Install dependencies
```bash
pip install pytest pandas matplotlib numpy
```

### Run tests
```bash
cd phase3
python -m pytest tests/test_scoring.py -v
```

Expected output: `54 passed in 0.08s`

### Use scoring module
```python
from scoring import assess_patient

# Patient input
questionnaire = {
    'A1': 3, 'A2': 3, 'A3': 2,    # TMJ pain
    'A4': 2, 'A5': 3, 'A6': 3,    # Dysfunction
    'B1': 3, 'B2': 2,              # Bruxism
    'C1': 3, 'C2': 2,              # Stress (conditional)
    'E1': 2, 'E3': 1,              # Demographics (31-45 female)
}

# Image detections from Phase 1 YOLO model
detections = [
    {'class': 'D5_distinct_cavity', 'confidence': 0.88, 
     'bbox': [0.15, 0.2, 0.25, 0.3]},
]

# Assessment
result = assess_patient(questionnaire, detections)

print(result['tmj_risk']['category'])     # 'HIGH'
print(result['tmj_risk']['action_id'])    # 'Konsultasi minggu ini'
print(result['caries']['category'])        # 'URGENT'
```

## 📊 Core Components

### 1. Questionnaire Scorer (`questionnaire_scorer.py`)

6 sub-score functions, each validated against DC/TMD literature:

| Function | Input | Output Range | Source |
|----------|-------|--------------|--------|
| `calculate_tmj_pain_subscore` | A1, A2, A3 | 0-100 | TMD Pain Screener |
| `calculate_tmj_dysfunction_subscore` | A4, A5, A6 | 0-100 | TMD Pain Screener |
| `calculate_bruxism_subscore` | B1, B2 | 0-100 | DC/TMD Axis I |
| `calculate_stress_modifier` | C1, C2, C3 | 1.0-1.4 | PHQ-4 adapted |
| `calculate_trauma_score` | D1 | 0-15 | Clinical heuristic |
| `calculate_demographics_modifier` | E1, E3 | 0.7-1.8 | Slade et al. 2013 |

### 2. Image Feature Extractor (`image_feature_extractor.py`)

Converts YOLO Phase 1 output to tabular features:
- Per-class detection counts (D0-D6)
- Total caries, severe caries (D5+D6)
- Wear pattern score (D3+D4 weighted)
- Left-right asymmetry score
- Average confidence

### 3. Triage Mapping (`triage.py`)

**TMJ Risk Triage:**
- `LOW` (0-30): Routine 6-month check
- `MODERATE` (31-60): Konsultasi 1-2 minggu
- `HIGH` (61-100): Urgent this week

**Caries Triage:**
- `URGENT`: Any severe caries (D5/D6)
- `SOON`: Multiple moderate (D3/D4 ≥ 2) or multiple early (D1/D2 ≥ 3)
- `ROUTINE`: No significant caries

### 4. Risk Calculator (`risk_calculator.py`)

Master function `assess_patient()`:
- Calls all sub-scorers + image extractor
- Combines with weighted formula
- Returns comprehensive dict with TMJ + caries triage + sub-scores + disclaimer

**Component weights (v0.1, hand-tuned):**
- Pain: 30%
- Dysfunction: 30%
- Bruxism: 20%
- Image: 10%
- Trauma: 10%
- Modifiers: stress (1.0-1.4x) × demographics (0.7-1.8x)

> ⚠️ **Note:** Weights akan di-learn XGBoost di Phase 3.4 (replace hand-tuned).

## 🧪 Test Coverage

54 tests covering:

- **Boundary cases**: all 0s (sehat), all 4s (severe), missing items
- **Validation errors**: out-of-range values, missing required items
- **Sanity checks**: severe > healthy, score in [0-100] range
- **Conditional logic**: Section C/D missing returns sensible defaults
- **Image feature extraction**: empty input, asymmetric, severe caries
- **Triage boundaries**: 30/31, 60/61 transitions
- **End-to-end**: 5 personas covering range of cases

## 📈 Validated Results (Smoke Test)

| Patient Profile | TMJ Score | Triage | Caries |
|----------------|-----------|--------|--------|
| Sehat (all 0s) | 0.00 | LOW | ROUTINE |
| Moderate symptoms + stress | ~73 | HIGH | depends on image |
| Severe (all 4s) + female | 100 | HIGH | URGENT |

## 🔬 Known Limitations

1. **Hand-tuned weights**: v0.1 weights are heuristic. Phase 3.4 will replace with XGBoost-learned weights.

2. **Modifier amplification**: Stress + Demographics multiplicative can amplify base score 2.5x. Need clinical validation.

3. **Image feature heuristics**: Wear pattern & asymmetry detection are simple counting/positional. Future: dedicated wear detector model.

4. **No real validation**: Synthetic data validation only. Real validation post-clinic launch.

## 📚 References

1. Schiffman E, et al. (2014). *J Oral Facial Pain Headache*. 28(1):6-27.
2. Gonzalez YM, Schiffman E. (2011). *J Am Dent Assoc*. 142(10):1183-91.
3. Slade GD, et al. (2013). *J Dent Res*. 95(10):1084-1092.
4. Van der Meer HA, et al. (2021). *J Oral Facial Pain Headache*. 35(2):150-156.

## ⏭️ Next: Phase 3.3

Generate synthetic patient dataset (~5000 profiles) for XGBoost training:
- Realistic prevalence distribution (TMD 5-30% global)
- Co-morbidity patterns (bruxism+stress, trauma+chronic)
- Indonesian demographic reference
