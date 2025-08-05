import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("")

features = [
    "Chest_Pain","Shortness_of_Breath","Fatigue","Palpitations",
    "Dizziness","Swelling","Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea","High_BP","High_Cholesterol",
    "Diabetes","Smoking","Obesity","Sedentary_Lifestyle",
    "Family_History","Chronic_Stress","Gender","Age",
]

df["growth"] = (df["Heart_Risk"] == 1)

x = df[features]
y = df["Growth"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = ImbPipeline(steps=[('classifier', RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_leaf=250,
        random_state=42,
        n_jobs=-1
    ))])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
y_proba = pipeline.predict_proba(x_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.2f}")
