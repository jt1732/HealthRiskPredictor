from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from data import RANDOM_STATE
from train import METRICS, MODELS, load_training_data, make_pipeline

BEST_MODEL_PATH = Path(__file__).parent / "best_model.joblib"

SEARCH_SPACES = {
    "logistic regression": {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
    "decision tree": {
        "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 5, 10, 20, 50, 100],
    },
    "random forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_leaf": [1, 5, 10, 20, 50],
        "model__max_features": ["sqrt", 0.5, None],
    },
    "hist gradient boosting": {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 400],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 50, 100],
        "model__l2_regularization": [0, 0.1, 1],
    },
    "naive bayes": {
        "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5, 1e-3],
    },
}


if __name__ == "__main__":
    X, y = load_training_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_params = {}
    searches = {}
    for name, params in SEARCH_SPACES.items():
        print(f"Tuning {name}...")
        search = RandomizedSearchCV(
            make_pipeline(MODELS[name]),
            param_distributions=params,
            n_iter=20,  
            scoring=METRICS,
            refit="roc_auc", 
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X, y)

        best = search.best_index_
        results[name] = {metric: search.cv_results_[f"mean_test_{metric}"][best] for metric in METRICS}
        best_params[name] = search.best_params_
        searches[name] = search

    print("\nTuned models, mean score over 5 folds:")
    print(pd.DataFrame(results).T.round(3).to_string())

    print("\nBest hyperparameters:")
    for name, params in best_params.items():
        print(f"  {name}: {params}")

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    joblib.dump(searches[best_name].best_estimator_, BEST_MODEL_PATH)
    print(f"\nBest model: {best_name} (ROC-AUC {results[best_name]['roc_auc']:.3f}), saved to {BEST_MODEL_PATH.name}")
