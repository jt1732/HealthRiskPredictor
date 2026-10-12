import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data import RANDOM_STATE, TARGET, split_data
from preprocessing import add_features, build_preprocessor, remove_invalid_rows

METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "sensitivity": "recall",
    "specificity": make_scorer(recall_score, pos_label=0),
    "f1": "f1",
    "roc_auc": "roc_auc",
}

def load_training_data():
    train_df, _ = split_data()
    train_df = add_features(remove_invalid_rows(train_df))
    X = train_df.drop(columns=TARGET)
    y = train_df[TARGET]
    return X, y

def make_pipeline(model):
   return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])

MODELS = {
    "dummy baseline": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=1000),
    "decision tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE),
    "hist gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "naive bayes": GaussianNB(),
}

if __name__ == "__main__":
    X, y = load_training_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, model in MODELS.items():
        print(f"Cross-validating {name}...")
        scores = cross_validate(make_pipeline(model), X, y, cv=cv, scoring=METRICS)
        results[name] = {metric: scores[f"test_{metric}"].mean() for metric in METRICS}

    print("\nMean score over 5 folds:")
    print(pd.DataFrame(results).T.round(3).to_string())
