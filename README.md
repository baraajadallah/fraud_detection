# fraud_detection

## 1) Introduction

Data: 594,643 transactions with columns like age_group, gender_clean, category_clean, amount_bin, merchant, customer, fraud (1/0).

Challenge: Heavy class imbalance (~1.2% fraud), high-cardinality IDs (merchant, customer), and strong amount effects.

Goals: High precision/recall with stable behavior across spend bands (amount_bin).

## 2) Models at a Glance

| Model                                   | PR-AUC |  ROC-AUC | Test Precision | Test Recall | Threshold Policy                   |
| --------------------------------------- | -----: | -------: | -------------: | ----------: | ---------------------------------- |
| **Random Forest** (OHE + TE + CountEnc) | \~0.83 | \~0.9967 |         \~0.78 |      \~0.74 | Global + per-bin (optional)        |
| **CatBoost** (native categorical)       | \~0.93 | \~0.9984 |         \~0.85 |      \~0.85 | **Global + per-bin (recommended)** |

Notes: CatBoost consistently improves mid/low amount_bin performance. Per-bin thresholds (learned on validation) boost recall in small-amount bins without hurting high-amount performance.

## 4) Inference (How to Use)

Batch scoring on new data:

Input must include either amount_bin or raw amount (pipeline bins using training cutpoints).

Required columns: age_group, gender_clean, category_clean, amount_bin, merchant, customer.

Load artifacts:

CatBoost model: models/catboost_fraud_model.cbm

Threshold policy: models/thresholds_by_amount_bin.json (+ models/global_threshold.txt)

Compute predict_proba → apply per-bin threshold (fallback to global) → output proba_fraud, threshold_used, pred_fraud.

See notebooks/03_inference_catboost.ipynb (and RF equivalent) for a ready-to-run pipeline.

## Quick Set-up

pip install pandas numpy scikit-learn catboost joblib matplotlib seaborn
add your csv file locally; run training/inference notebooks as needed

