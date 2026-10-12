# Heart Risk Predictor v2

A rebuild of the v1 heart-risk classifier as a proper ML pipeline on the Cardiovascular Disease dataset (`cardio.csv`, 70,000 patients).
v1 had no baseline, no proper evaluation and no real preprocessing. v2 fixes all three.

## Results

Best model: **HistGradientBoostingClassifier**, tuned with 5-fold cross-validation and evaluated once on the held-out test set.

| Metric | Dummy baseline (CV) | Best model (CV) | Best model (test set) |
|--------|---------------------|-----------------|------------------------|
| accuracy | 0.505 | 0.735 | **0.733** |
| precision | 0.000 | 0.753 | **0.747** |
| sensitivity | 0.000 | 0.692 | **0.697** |
| specificity | 1.000 | 0.777 | **0.769** |
| f1 | 0.000 | 0.721 | **0.721** |
| roc_auc | 0.500 | 0.802 | **0.798** |

- The test scores are within 0.01 of the CV scores, so the model isn't overfitting and the CV estimate was reliable
- Every model (logistic regression, decision tree, random forest, gradient boosting) scored between 0.79 and 0.80 ROC-AUC, and tuning added at most 0.001. The limit is the information in the features, not the choice of model
- The model misses about 30% of sick patients (2,058 of 6,784 in the test set). For screening, the decision threshold could be lowered to catch more, at the cost of more false alarms

## How to run

From the repository root:

```
pip install -r requirements.txt
python modelv2/data.py          # load and split, prints class balance
python modelv2/preprocessing.py # cleaning counts and scaling check
python modelv2/train.py         # 5-fold CV comparison of all models (about 30 seconds)
python modelv2/tune.py          # hyperparameter search, saves best_model.joblib (about 5 minutes)
python modelv2/evaluate.py      # final test-set evaluation
```

`eda.ipynb` contains the exploratory data analysis.

## Files

| File | Purpose |
|------|---------|
| `data.py` | Load `cardio.csv` and make a stratified 80/20 train/test split |
| `eda.ipynb` | EDA on the training set only |
| `CLEANING_PLAN.md` | Problems found in the EDA and the rules to fix them |
| `preprocessing.py` | `remove_invalid_rows`, `add_features` and the `ColumnTransformer` |
| `train.py` | Cross-validated comparison of all models |
| `tune.py` | `RandomizedSearchCV` for every model, saves the best one |
| `evaluate.py` | One evaluation of the best model on the test set |
| `best_model.joblib` | The trained best pipeline (preprocessing + model) |

## Key decisions

- **Split first.** The stratified train/test split happens before any cleaning or EDA, so the test set never influences a decision
- **Fixed cleaning rules, not statistical ones.** Rows are removed only when they are physically implausible (e.g. height outside 130 to 210 cm, ap_lo ≥ ap_hi). A p < 0.05 cut was rejected because it removes about 5% of real patients by design. Cleaning removes 2.1% of rows
- **Drop BP typing errors instead of repairing them.** Values like ap_lo = 1000 are probably 100 with an extra 0, but a repair is a guess
- **BMI added as a feature.** It combines height and weight and is a known risk factor
- **Scaling inside the pipeline.** The scaler is fitted on training data only, and cross-validation refits it on each fold, so there is no leakage
- **No SMOTE or class weights.** The target is balanced (50/50)
- **Metrics beyond accuracy.** The dummy baseline scores 50.5% accuracy while catching no sick patients, which is why sensitivity, specificity and ROC-AUC are reported
- **Model chosen by ROC-AUC**, because it measures ranking quality independently of the 0.5 threshold
