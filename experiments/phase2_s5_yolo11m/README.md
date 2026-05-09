\# Phase 2 — Scenario S5: YOLOv11m (Capacity Scaling)



\*\*Date:\*\* 2026-05-05  

\*\*Status:\*\* ❌ Did not improve baseline  

\*\*Test mAP@0.5:\*\* 0.6389 (vs Phase 1: 0.6492, \*\*-1.6%\*\*)



> ⚠️ Note: Output files (best.pt, plots) lost due to Kaggle session crash.

> Metrics captured in this README from training logs.



\## Hypothesis



YOLOv11m (20.1M params, 2.1x capacity vs YOLOv11s) akan belajar fitur halus

untuk improve D2 dan reduce class confusion.



\## Result: Hypothesis REJECTED



| Metric | Phase 1 (s) | S5 (m) | Δ |

|--------|-------------|--------|---|

| mAP@0.5 | 0.6492 | 0.6389 | \*\*-0.0103\*\* |

| mAP@0.5:0.95 | 0.6170 | 0.6095 | -0.0075 |

| Precision | 0.5138 | 0.5179 | +0.0041 |

| Recall | 0.7519 | 0.7336 | -0.0183 |



\## Per-Class Breakdown



| Class | Phase 1 | S5 | Δ | Notes |

|-------|---------|-----|---|-------|

| D0\_sound | 0.7822 | 0.7788 | -0.0034 | |

| D1\_first\_change | 0.4725 | 0.4862 | \*\*+0.0137\*\* | Only class with improvement |

| D2\_distinct\_opacity | 0.3324 | 0.3199 | -0.0125 | Still weakest |

| D3\_enamel\_breakdown | 0.7065 | 0.6926 | -0.0139 | |

| D4\_dentin\_shadow | 0.6122 | 0.6049 | -0.0073 | |

| D5\_distinct\_cavity | 0.7556 | 0.7497 | -0.0059 | |

| D6\_extensive\_cavity | 0.6576 | 0.6343 | \*\*-0.0233\*\* | Worst regression |



\## Overfitting Health Check



\- Val mAP@0.5: 0.6472

\- Test mAP@0.5: 0.6389

\- Gap: +0.0082 (healthy, similar to Phase 1's +0.0085)



\*\*Verdict: NOT overfitting.\*\* Model size 2x didn't cause memorization.



\## Why It Failed



1\. \*\*Capacity-absorption phenomenon\*\*: Extra capacity dipakai model untuk

&#x20;  refine majority classes (D1) sambil mengorbankan rare classes (D6 -2.3%).

2\. \*\*Dataset already saturated\*\*: 4,617 images sufficient for YOLOv11s.

&#x20;  Bigger model needs proportionally more data.

3\. \*\*Inherent class similarity\*\*: D2 (mid-spectrum opacity) is anatomically

&#x20;  ambiguous — even radiologists have inter-observer disagreement.



\## Strategic Insight (Most Important)



This was the 2nd experiment confirming \*\*model-centric plateau\*\*:

\- Phase 1 (baseline): 0.6492

\- S3 (heavy aug): 0.6345 

\- S5 (larger model): 0.6389



Bottleneck is \*\*data-centric\*\*, not model-centric. Continued tweaking

of architecture/hyperparams will yield diminishing returns.



\## Decision



\*\*Accept Phase 1 (mAP 0.65) as baseline final.\*\*

Pivot strategy from model-centric optimization to multimodal fusion + 

real-world deployment (Phase 3+).



Real model improvement deferred to post-launch when clinic data 

(from Sukabumi patients) becomes available for fine-tuning.



\## Configuration Reference



\- Model: yolo11m.pt (COCO pretrained)

\- Dataset: Roboflow dental-caries-7kttb v19

\- Epochs: 50

\- Batch: 16 (vs 32 in Phase 1, due to memory)

\- Image size: 640

\- Optimizer: AdamW, lr=0.001, cosine schedule

\- Augmentation: Default Ultralytics



\## Lost Artifacts



\- `best.pt` (38 MB) — not needed for downstream phases

\- `confusion\_matrix\_normalized.png`

\- `results.png` (training curves)

\- val\_batch predictions



These are documented in metrics tables above. Re-running not necessary

since Phase 1 weights are the baseline going forward.

