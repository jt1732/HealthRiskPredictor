import tkinter as tk
from tkinter import messagebox
import math
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
loaded_model = joblib.load('random_forest_model.joblib')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a simple GUI
root = tk.Tk()
root.title("Heart Risk Predictor")
frm = tk.Frame(root, padx=10, pady=10)
frm.grid()

widgets = {
    "Chest_Pain": ["experience chest pain"],
    "Shortness_of_Breath": ["experience shortness of breath"],
    "Fatigue": ["experience fatigue"],
    "Palpitations": ["experience palpitations"],
    "Dizziness": ["experience dizziness"],
    "Swelling": ["experience swelling"],
    "Pain_Arms_Jaw_Back": ["experience pain in your arms, jaw or back"],
    "Cold_Sweats_Nausea": ["experience cold sweats or nausea"],
    "High_BP": ["experience high BP"],
    "High_Cholesterol": ["have high cholesterol"],
    "Diabetes": ["have diabetes"],
    "Smoking": ["smoke"],
    "Obesity": ["have obesity"],
    "Sedentary_Lifestyle": ["have a sedentary lifestyle"],
    "Family_History": ["have a family history"],
    "Chronic_Stress": ["experience chronic stress"]
}

def create_widgets(lbl_name,lbl_text,row, column):
    lbl = tk.Label(frm, text="Do you " + lbl_text +"?")
    listbox = tk.Listbox(frm, name=lbl_name.lower(), width=5, height=2, exportselection=False)
    listbox.original_name = lbl_name
    listbox.insert(0,"No")
    listbox.insert(1,"Yes")
    lbl.grid(row=row, column=column)
    listbox.grid(row=row+1, column=column)
    return listbox

widget_complete = []
for i, (key, widget_info) in enumerate(widgets.items()):
    row = math.floor(i / 4) * 2
    col = i % 4
    widget_complete.append(create_widgets(key, widget_info[0], row, col))

def predict():
    dictData = {}
    for widget in widget_complete:
        if not widget.curselection():
            messagebox.showwarning("Warning", "Please fill in all options")
            return
        for i in widget.curselection():
            dictData[widget.original_name] = 1 if widget.get(i) == "Yes" else 0

    df = pd.DataFrame(dictData, index=[0])
    predictions = loaded_model.predict(df)
    probabilities = loaded_model.predict_proba(df)
    if predictions[0]:
        outcome = "likely to be at"
    else:
        outcome = "unlikely to be at"
    prob = round(probabilities[0, 1] * 100, 2)
    txtBox.delete("0.0", tk.END)
    txtBox.insert('0.0', f"Our model predicts that with {prob}% probability you are {outcome} heart risk")


txtBox = tk.Text(root, height=10, width =48)
btn = tk.Button(root,text="Click Me", command=predict)
txtBox.grid(pady=10)
btn.grid()
root.mainloop()