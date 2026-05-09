# SmileGuard Scoring Logic — Outline v0.1

> **Status:** Sketch / Outline untuk Phase 3.2 implementation  
> **Last update:** 2026-05-05  
> **Owner:** Riexu

## 🎯 Tujuan Dokumen

Dokumen ini menjelaskan **bagaimana** kita menghitung TMJ Risk Score dari kombinasi:
1. Jawaban kuesioner pasien (18 items, adaptive)
2. Output Phase 1 YOLO model (deteksi karies + indicators)

Ini **outline** — implementasi detail (kode Python + XGBoost training) dilakukan di Phase 3.2.

---

## 📊 Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT SOURCES                            │
├──────────────────────────────┬──────────────────────────────┤
│  Questionnaire (18 items)     │  Image Output (Phase 1 YOLO) │
│  - Sections A-E               │  - 7 ICDAS class detections   │
│  - Likert 5-point scale       │  - Bbox positions             │
│  - Adaptive (8 core + 10      │  - Confidence scores          │
│    conditional)               │                                │
└──────────────────────────────┴──────────────────────────────┘
                ↓                                ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                          │
├──────────────────────────────┬──────────────────────────────┤
│  Tabular features:            │  Image-derived features:      │
│  - tmj_pain_subscore          │  - n_caries_total             │
│  - tmj_dysfunction_subscore   │  - n_caries_by_severity       │
│  - bruxism_subscore           │  - dominant_severity_class    │
│  - stress_subscore            │  - wear_pattern_indicator     │
│  - trauma_score               │  - asymmetry_score            │
│  - demographics_modifier      │                                │
└──────────────────────────────┴──────────────────────────────┘
                ↓                                ↓
                └────────────┬───────────────────┘
                             ↓
            ┌────────────────────────────────────┐
            │      FUSION MODEL (XGBoost)        │
            │  Train: synthetic dataset (5000)   │
            │  Output: TMJ Risk Score 0-100      │
            └────────────────────────────────────┘
                             ↓
            ┌────────────────────────────────────┐
            │         TRIAGE MAPPING             │
            │  Low: 0-30                         │
            │  Moderate: 31-60                   │
            │  High: 61-100                      │
            └────────────────────────────────────┘
```

---

## 🧮 Sub-Score Formulas

### 1. TMJ Pain Sub-score (from Section A: A1, A2, A3)

```python
def calculate_tmj_pain_subscore(answers):
    """
    Pain-related TMD indicators.
    Range: 0-100
    """
    a1 = answers['A1']  # Pain in jaw/temple/ear
    a2 = answers['A2']  # Pain modified by movement
    a3 = answers['A3']  # Headache in temple area
    
    # Weights from clinical literature
    weighted_sum = (
        a1 * 3.5 +     # primary pain indicator
        a2 * 3.0 +     # functional pain
        a3 * 2.5       # referred pain
    )
    
    # Max possible: (4 * 3.5) + (4 * 3.0) + (4 * 2.5) = 36
    # Normalize to 0-100
    return (weighted_sum / 36) * 100
```

**Klinis reasoning:** A1 punya weight tertinggi (3.5) karena ini direct pain indicator. A2 (modified by movement) = strong TMJ-specific signal. A3 (headache) bisa sekunder tapi tetap relevan.

### 2. TMJ Dysfunction Sub-score (from Section A: A4, A5, A6)

```python
def calculate_tmj_dysfunction_subscore(answers):
    """
    Functional limitation & joint dysfunction indicators.
    Range: 0-100
    """
    a4 = answers['A4']  # Stiffness, locking
    a5 = answers['A5']  # Clicking, popping sounds
    a6 = answers['A6']  # Morning stiffness
    
    weighted_sum = (
        a4 * 3.5 +     # mechanical dysfunction (high signal)
        a5 * 2.5 +     # joint sound (moderate signal — bisa harmless)
        a6 * 3.0       # bruxism + TMJ overload indicator
    )
    
    return (weighted_sum / 36) * 100
```

**Klinis reasoning:** A4 (locking) = serious mechanical issue, weight tertinggi. A5 (clicking) lebih lemah karena banyak orang punya joint sound tanpa pathology. A6 (morning stiffness) = strong bruxism overload signal.

### 3. Bruxism Sub-score (from Section B: B1, B2)

```python
def calculate_bruxism_subscore(answers):
    """
    Sleep & awake bruxism indicators.
    Range: 0-100
    """
    b1 = answers['B1']  # Sleep bruxism
    b2 = answers['B2']  # Awake bruxism / clenching
    
    weighted_sum = (
        b1 * 3.0 +     # sleep bruxism — strong correlation with TMD
        b2 * 2.5       # awake bruxism — moderate
    )
    
    # Max: (4 * 3.0) + (4 * 2.5) = 22
    return (weighted_sum / 22) * 100
```

### 4. Stress Modifier (from Section C, conditional)

```python
def calculate_stress_modifier(answers):
    """
    Psychosocial modifier. Returns multiplier 1.0-1.4.
    Only applied if Section C was answered (conditional triggered).
    """
    if 'C1' not in answers:
        return 1.0  # No modifier
    
    stress_total = answers.get('C1', 0) + answers.get('C2', 0) + answers.get('C3', 0)
    # Max: 12
    
    # Linear scaling: 0 → 1.0, 12 → 1.4
    modifier = 1.0 + (stress_total / 12) * 0.4
    return modifier
```

**Klinis reasoning:** Stress tinggi = TMD severity meningkat ~20-40%. Bukan main driver, tapi multiplier yang relevan.

### 5. Trauma Score (from Section D, conditional)

```python
def calculate_trauma_score(answers):
    """
    Trauma history score. Returns 0-15 additive.
    More recent trauma = higher score.
    """
    if 'D1' not in answers:
        return 0
    
    d1 = answers['D1']
    weights = {0: 0, 1: 1, 2: 3, 3: 8, 4: 15}  # exponential for recent
    return weights[d1]
```

### 6. Demographics Modifier

```python
def calculate_demographics_modifier(answers):
    """
    Age & gender adjustment.
    Returns multiplier 0.7-1.8.
    """
    age = answers['E1']
    gender = answers['E3']
    
    age_modifier = {
        0: 0.7,   # < 18 (juvenile, lower TMD prevalence)
        1: 1.0,   # 18-30 (peak TMD onset)
        2: 1.2,   # 31-45 (peak TMD prevalence)
        3: 1.0,   # 46-60
        4: 0.8    # > 60 (different etiology, mostly degenerative)
    }
    
    gender_modifier = {
        0: 1.0,   # Male
        1: 1.5,   # Female (2-3x higher prevalence)
        2: 1.0    # Not specified
    }
    
    return age_modifier[age] * gender_modifier[gender]
```

**Citation:** Slade et al. 2013 — TMD prevalence ratio female:male = 2:1, peak age 20-40.

---

## 🎨 Image-Derived Features (dari Phase 1 YOLO)

Output Phase 1 model = list of detections per image. Kita extract features:

```python
def extract_image_features(yolo_detections):
    """
    Convert YOLO output → tabular features for fusion model.
    
    yolo_detections: list of {bbox, class, confidence}
    """
    features = {}
    
    # Caries severity distribution
    severity_classes = ['D1_first_change', 'D2_distinct_opacity', 
                        'D3_enamel_breakdown', 'D4_dentin_shadow',
                        'D5_distinct_cavity', 'D6_extensive_cavity']
    
    for cls in severity_classes:
        features[f'n_{cls}'] = sum(1 for d in yolo_detections if d['class'] == cls)
    
    # Aggregate features
    features['n_caries_total'] = sum(features[f'n_{c}'] for c in severity_classes)
    
    # Severe caries (D5 + D6) — strong signal for urgency
    features['n_severe_caries'] = (
        features.get('n_D5_distinct_cavity', 0) + 
        features.get('n_D6_extensive_cavity', 0)
    )
    
    # Wear pattern indicator (from D3-D4 distribution)
    # If many D3-D4 across multiple teeth = bruxism wear pattern
    features['wear_pattern_score'] = (
        features.get('n_D3_enamel_breakdown', 0) + 
        features.get('n_D4_dentin_shadow', 0) * 1.5
    )
    
    # Asymmetry score (proxy from bbox positions)
    # If caries cluster on one side only → asymmetric mastication
    features['asymmetry_score'] = calculate_lr_asymmetry(yolo_detections)
    
    return features


def calculate_lr_asymmetry(detections):
    """
    Compare caries distribution left vs right (bbox x_center < 0.5 vs >= 0.5).
    Returns 0 (perfectly symmetric) to 1 (all on one side).
    """
    if not detections:
        return 0
    
    left = sum(1 for d in detections if d['bbox'][0] < 0.5)
    right = sum(1 for d in detections if d['bbox'][0] >= 0.5)
    total = left + right
    
    if total == 0:
        return 0
    
    return abs(left - right) / total
```

---

## 🔮 Final TMJ Risk Score Formula

```python
def calculate_tmj_risk_score(questionnaire_answers, image_features):
    """
    Master scoring function.
    Returns: float in [0, 100]
    """
    # === Sub-scores from questionnaire ===
    pain_score = calculate_tmj_pain_subscore(questionnaire_answers)
    dysfunction_score = calculate_tmj_dysfunction_subscore(questionnaire_answers)
    bruxism_score = calculate_bruxism_subscore(questionnaire_answers)
    
    # === Modifiers ===
    stress_mod = calculate_stress_modifier(questionnaire_answers)
    trauma_score = calculate_trauma_score(questionnaire_answers)
    demo_mod = calculate_demographics_modifier(questionnaire_answers)
    
    # === Image contribution ===
    image_contrib = (
        image_features['wear_pattern_score'] * 2 +
        image_features['asymmetry_score'] * 10
    )
    
    # === Weighted combination ===
    base_score = (
        pain_score * 0.30 +           # 30% weight
        dysfunction_score * 0.30 +     # 30% weight
        bruxism_score * 0.20 +         # 20% weight
        image_contrib * 0.10 +         # 10% weight
        trauma_score * 0.10            # 10% weight
    )
    
    # Apply modifiers (multiplicative)
    final_score = base_score * stress_mod * demo_mod
    
    # Clamp to [0, 100]
    return max(0, min(100, final_score))
```

**⚠️ NOTE:** Weights di atas adalah **starting point**. Di Phase 3.4, kita akan **train XGBoost** untuk learn weights optimal dari synthetic dataset, bukan hand-tuned.

---

## 🚦 Triage Mapping

```python
def get_triage_category(risk_score):
    """Map score to action recommendation."""
    if risk_score < 31:
        return {
            'category': 'LOW',
            'category_id': 'Risiko Rendah',
            'action_id': 'Periksa rutin (6 bulan)',
            'urgency_id': 'Tidak mendesak',
            'color': 'green'
        }
    elif risk_score < 61:
        return {
            'category': 'MODERATE',
            'category_id': 'Risiko Sedang',
            'action_id': 'Konsultasi dalam 1-2 minggu',
            'urgency_id': 'Disarankan segera',
            'color': 'yellow'
        }
    else:
        return {
            'category': 'HIGH',
            'category_id': 'Risiko Tinggi',
            'action_id': 'Konsultasi minggu ini',
            'urgency_id': 'Perlu perhatian segera',
            'color': 'red'
        }
```

---

## 🎯 Caries Triage (Separate from TMJ)

Caries punya triage sendiri based on Phase 1 detections:

```python
def get_caries_triage(image_features):
    n_severe = image_features['n_severe_caries']  # D5 + D6
    n_moderate = image_features.get('n_D3_enamel_breakdown', 0) + \
                 image_features.get('n_D4_dentin_shadow', 0)
    n_early = image_features.get('n_D1_first_change', 0) + \
              image_features.get('n_D2_distinct_opacity', 0)
    
    if n_severe >= 1:
        return 'URGENT'      # Any severe = urgent
    elif n_moderate >= 2:
        return 'SOON'         # Multiple moderate = soon
    elif n_early >= 3:
        return 'SOON'         # Multiple early = monitor
    else:
        return 'ROUTINE'
```

---

## ⚠️ Limitations & Caveats

1. **Initial weights = sketch.** Real weights akan di-learn dari training data di Phase 3.4.

2. **No real validation yet.** Scoring belum divalidasi dengan ground truth. Validation post-launch via paman's clinical observations.

3. **Synthetic data risk.** Phase 3.3 akan generate synthetic patient profiles. Kalau distribusi tidak match realita Sukabumi, model akan biased.

4. **Image features simplified.** Wear pattern & asymmetry detection saat ini heuristic. Future improvement: dedicated ML model untuk wear detection.

5. **Cultural context.** Pertanyaan tentang stres atau psychosocial mungkin under-reported di konteks Indonesia karena stigma. Threshold mungkin perlu adjusted.

---

## 📋 TODO untuk Phase 3.2-3.6

- [ ] **3.2:** Implement scoring functions ini sebagai Python module
- [ ] **3.3:** Generate synthetic patient dataset (~5000 profiles) using clinical rules
- [ ] **3.4:** Train XGBoost untuk learn optimal weights
- [ ] **3.5:** Calibration & evaluation
- [ ] **3.6:** Integration test end-to-end pipeline
- [ ] **Post-launch:** Collect real patient data → fine-tune weights

---

## 📚 References

1. Schiffman E, Ohrbach R, Truelove E, et al. (2014). "Diagnostic Criteria for Temporomandibular Disorders (DC/TMD) for Clinical and Research Applications." *J Oral Facial Pain Headache*. 28(1):6-27.

2. Gonzalez YM, Schiffman E, et al. (2011). "Development of a brief and effective temporomandibular disorder pain screening questionnaire." *J Am Dent Assoc*. 142(10):1183-91.

3. Slade GD, et al. (2013). "Painful Temporomandibular Disorder: Decade of Discovery from OPPERA Studies." *J Dent Res*. 95(10):1084-1092.

4. Van der Meer HA, et al. (2021). "Validation of the Temporomandibular Disorder Pain Screener in a Specialized Headache Center." *J Oral Facial Pain Headache*. 35(2):150-156.

5. Durham J, et al. (2024). "Constructing the brief diagnostic criteria for temporomandibular disorders (bDC/TMD) for field testing." *J Oral Rehabil*. 51:785-794.
