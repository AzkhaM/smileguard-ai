# \# Phase 1 — Baseline YOLOv11s

# 

# \*\*Date:\*\* 2026-05-05

# \*\*Status:\*\* ✅ Complete

# \*\*Duration:\*\* \~40 minutes (50 epochs, T4 x2)

# 

# \## Configuration

# 

# | Parameter | Value |

# |-----------|-------|

# | Model | YOLOv11s (pretrained COCO) |

# | Dataset | Roboflow dental-caries-7kttb v19 |

# | Train/Val/Test | 3,228 / 939 / 450 images |

# | Classes | 7 (ICDAS D0–D6) |

# | Epochs | 50 |

# | Batch size | 32 |

# | Image size | 640 |

# | Optimizer | AdamW, lr=0.001, cosine schedule |

# | Augmentation | Default Ultralytics |

# 

# \## Test Metrics

# 

# | Metric | Value |

# |--------|-------|

# | mAP@0.5 | \*\*0.6492\*\* |

# | mAP@0.5:0.95 | 0.6170 |

# | Precision | 0.5138 |

# | Recall | 0.7519 |

# 

# \## Per-Class mAP@0.5 (Test Set)

# 

# | Class | mAP@0.5 | Notes |

# |-------|---------|-------|

# | D0\_sound | 0.7822 | 🥇 Top |

# | D5\_distinct\_cavity | 0.7556 | Strong despite low samples |

# | D3\_enamel\_breakdown | 0.7065 | Solid |

# | D6\_extensive\_cavity | 0.6576 | Strong despite rare class |

# | D4\_dentin\_shadow | 0.6122 | Decent |

# | D1\_first\_change | 0.4725 | ⚠️ Underperform |

# | D2\_distinct\_opacity | 0.3324 | 🚨 Weakest |

# 

# \## Key Findings

# 

# 1\. \*\*Severe cases (D5/D6) detected accurately\*\* — critical for triage urgency

# 2\. \*\*D2 weakest class\*\* — likely confused with D1/D3 (visually similar transition stages)

# 3\. \*\*Train/test consistency excellent\*\* — no overfitting (val mAP 0.658 vs test 0.649)

# 4\. \*\*Recall > Precision\*\* — model "rajin" detect, banyak false positives

# 

# \## Issues Identified for Phase 2

# 

# \- \*\*NMS aggressive overlap\*\* in predictions (multiple bbox per tooth)

# \- \*\*Class imbalance\*\*: D6 (1.08%) vs D0 (33.41%)

# \- \*\*D1↔D2↔D3 confusion\*\* in early-caries spectrum

# 

# \## Model Weights

# 

# Best weights tersedia di Hugging Face:

# 🤗 https://huggingface.co/<username>/smileguard-yolo/tree/main/phase1

# 

# ```python

# from ultralytics import YOLO

# \# Download dari HF Hub

# model = YOLO('phase1/best.pt')  # setelah download

# ```

