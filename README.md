# Fraud detection scoring (end-to-end)

End-to-end ML project on a highly imbalanced scoring task: **detecting fraudulent card transactions**.

The main deliverable is the notebook `notebooks/01_EDA.ipynb`, written as a reproducible workflow:

- quick EDA + class imbalance analysis
- preprocessing (scaling `Time`/`Amount`)
- baselines (LogReg) and model comparison (RandomForest, XGBoost)
- Optuna tuning (with `xgboost.train()` due to XGBoost 3.x sklearn API changes)
- business layer: Precision@K / Recall@K, cost-based threshold selection, virtual A/B on holdout
- StratifiedKFold cross-validation (ROC-AUC + PR-AUC)

## Dataset

Kaggle: Credit Card Fraud Detection.

Expected file: `data/creditcard.csv`.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open: `notebooks/01_EDA.ipynb`.

## Notes

- The preprocessing cell is idempotent (safe to rerun).
- With `xgboost==3.x`, early stopping in the sklearn wrapper `.fit()` is limited; Optuna tuning uses the native training API.

## Results (example)

From the current notebook run (seed=42):

- Tuned XGBoost: ROC-AUC ≈ 0.9733, PR-AUC ≈ 0.8759
- 5-fold CV: XGBoost baseline PR-AUC mean ≈ 0.8655 (see notebook for full table)
