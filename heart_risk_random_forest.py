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
