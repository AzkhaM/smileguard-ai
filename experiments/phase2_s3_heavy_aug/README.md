\# Phase 2 — Scenario S3: Heavy Augmentation



\*\*Date:\*\* 2026-05-05  

\*\*Status:\*\* ❌ Did not improve baseline  

\*\*Test mAP@0.5:\*\* 0.6345 (vs Phase 1: 0.6492, \*\*-1.5%\*\*)



\## Hypothesis



Heavy augmentation (mosaic, mixup, copy-paste, erasing, HSV) akan reduce 

background confusion (30%→<20%) dan improve D2 mAP (0.33→0.40+).



\## Result: Hypothesis REJECTED



| Metric | Phase 1 | S3 | Δ |

|--------|---------|-----|---|

| mAP@0.5 | 0.6492 | 0.6345 | \*\*-0.0147\*\* |

| Precision | 0.5138 | 0.5564 | +0.0426 |

| Recall | 0.7519 | 0.7004 | \*\*-0.0515\*\* |



\## Why It Failed



1\. \*\*Heavy augmentation made model overly conservative\*\* — recall dropped

2\. \*\*Domain-inappropriate augmentations\*\* — mixup, copy\_paste, erasing 

&#x20;  merusak struktur anatomical \& label integrity

3\. \*\*Continue training from converged Phase 1 weights\*\* dengan setup 

&#x20;  training drastically different = model needed to "un-learn"

4\. \*\*Dataset 4,617 images cukup besar\*\* → tidak butuh aug aggressive



\## Lessons Learned



\- Default YOLOv11 augmentation sudah include moderate aug yang cukup

\- Augmentation harus respect domain anatomy (no random erasing of caries area)

\- Negative results are valid scientific outcomes

\- Recall is more critical than precision for screening tools



\## Decision



Pivot strategy → focus on model capacity (YOLOv11m) instead of augmentation tweaking.

