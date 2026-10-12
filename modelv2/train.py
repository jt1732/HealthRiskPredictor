from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from data import RANDOM_STATE, TARGET, split_data
from preprocessing import add_features, build_preprocessor, remove_invalid_rows

# name shown in the output: how scikit-learn should calculate it
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


if __name__ == "__main__":
    X, y = load_training_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipeline = make_pipeline(DummyClassifier(strategy="most_frequent"))
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=METRICS)

    print("\nDummy baseline (5-fold CV):")
    for metric in METRICS:
        fold_scores = scores[f"test_{metric}"]
        print(f"  {metric}: {fold_scores.mean():.3f} +/- {fold_scores.std():.3f}")
