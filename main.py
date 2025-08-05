import tkinter as tk
import math
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a simple GUI
root = tk.Tk()
root.title("Heart Risk Predictor")
frm = tk.Frame(root, padx=10, pady=10)
frm.grid()

widgets = {
    "chest_pain": ["experience chest pain"],
    "shortness_of_breath": ["experience shortness of breath"],
    "fatigue": ["experience fatigue"],
    "palpitations": ["experience palpitations"],
    "dizziness": ["experience dizziness"],
    "swelling": ["experience swelling"],
    "arm_jaw_back_pain": ["experience pain in your arms, jaw or back"],
    "cold_sweats_nausea": ["experience cold sweats or nausea"],
    "high_bp": ["experience high BP"],
    "high_cholesterol": ["have high cholesterol"],
    "diabetes": ["have diabetes"],
    "smoking": ["smoke"],
    "obesity": ["have obesity"],
    "sedentary_lifestyle": ["have a sedentary lifestyle"],
    "family_history": ["have a family history"],
    "chronic_stress": ["experience chronic stress"]
}

def create_widgets(lbl_name,lbl_text,row, column):
    lbl = tk.Label(frm, text="Do you " + lbl_text +"?")
    listbox = tk.Listbox(frm, name=lbl_name, width=5, height=2, exportselection=False)
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

root.mainloop()