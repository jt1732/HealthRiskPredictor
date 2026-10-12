import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import TARGET, split_data
from preprocessing import add_features, remove_invalid_rows
from tune import BEST_MODEL_PATH

if __name__ == "__main__":
    _, test_df = split_data()
    test_df = add_features(remove_invalid_rows(test_df))
    X_test = test_df.drop(columns=TARGET)
    y_test = test_df[TARGET]

    model = joblib.load(BEST_MODEL_PATH) 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\nModel: {model.named_steps['model'].__class__.__name__}")
    print(f"Test set: {len(y_test)} patients\n")
    print(f"accuracy:    {accuracy_score(y_test, y_pred):.3f}")
    print(f"precision:   {precision_score(y_test, y_pred):.3f}")
    print(f"sensitivity: {recall_score(y_test, y_pred):.3f}")
    print(f"specificity: {tn / (tn + fp):.3f}")
    print(f"f1:          {f1_score(y_test, y_pred):.3f}")
    print(f"roc_auc:     {roc_auc_score(y_test, y_prob):.3f}")

    print("\nConfusion matrix:")
    print(f"                    predicted healthy  predicted disease")
    print(f"actually healthy    {tn:17d}  {fp:17d}")
    print(f"actually disease    {fn:17d}  {tp:17d}")
