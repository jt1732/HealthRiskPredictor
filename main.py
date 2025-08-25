import tkinter as tk
from http.client import responses
from tkinter import messagebox
import math
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
from fontTools.ttLib.tables.ttProgram import instructions
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

def create_widgets(lbl_name,lbl_text, __row, column):
    lbl = tk.Label(frm, text="Do you " + lbl_text +"?")
    listbox = tk.Listbox(frm, name=lbl_name.lower(), width=5, height=2, exportselection=False)
    listbox.original_name = lbl_name
    listbox.insert(0,"No")
    listbox.insert(1,"Yes")
    lbl.grid(row=__row, column=column)
    listbox.grid(row=__row+1, column=column)
    return listbox

def predict():
    dict_data = {}
    for widget in widget_complete:
        if not widget.curselection():
            messagebox.showwarning("Warning", "Please fill in all options")
            return
        else:
            dict_data[widget.original_name] = 1 if widget.get(0) == "Yes" else 0

    df = pd.DataFrame(dict_data, index=[0])
    prediction = loaded_model.predict(df)
    prob = round(loaded_model.predict_proba(df)[0, 1] * 100, 2)

    return prediction, prob, df

def display(predictions, prob, df):
    txtBox.delete("0.0", tk.END)
    outputText = accessOpenAI(client, prob, predictions, df)
    txtBox.insert(tk.INSERT, outputText)

def accessOpenAI(client, prob, predictions, df):
    __messages = [
        {
            "role": "system",
            "content": ("You are HeartGuide, a compassionate and informative AI assistant "
                        "for a cardiovascular health prediction app. Your purpose is to "
                        "explain medical data in a clear, reassuring, and non-alarming manner."
                        " You provide users with their prediction results, summarize their input,"
                        " and offer personalized, actionable feedback. You are not a doctor, but"
                        " a supportive guide. Always emphasize that this is a predictive model,"
                        " not a medical diagnosis, and strongly encourage them to share the results"
                        " with a healthcare professional.")
        },
        {
            "role": "user",
            "content": (
                f"Hello, HeartGuide. Please generate a personalized response for the user based on their heart health quiz results.\n\n"
                f"**MODEL OUTPUT:**\n"
                f"- Final Prediction: {predictions[0]}\n"
                f"- Probability of cardiovascular disease: {prob}%\n\n"
                f"**USER'S QUIZ ANSWERS:**\n"
                f"{df.to_string()}\n\n"  # Using to_string() for a better table format
                f"INSTRUCTIONS FOR YOUR RESPONSE:\n"
                f"Greet the user personally and state the purpose of the message.\n"
                f"Clearly state the Prediction and Probability. Frame it understandably (e.g., 'Based on your inputs, our model indicates a {prob}% probability of having cardiovascular disease.').\n"
                 f"Provide Custom Feedback: Analyze their specific answers. Pick the 2-3 most significant risk factors from their quiz answers and explain them in simple terms.\n"
                f"Offer Actionable Next Steps: Provide tailored advice based on their risk level.\n"
                f"Conclude with a Reassuring Disclaimer to see a doctor."
                f"Complete this in a concise manner with 175 words or less."
            )
        }
    ]
    __model="gpt-5-nano"
    response = client.responses.create(model=__model, input=__messages)
    print(response.output_text)
    return response.output_text

def submit():
    try:
        predictions, prob, df = predict()
        display(predictions, prob, df)
    except TypeError:
        print("Incorrect input")

widget_complete = []
for i, (key, widget_info) in enumerate(widgets.items()):
    row = math.floor(i / 4) * 2
    col = i % 4
    widget_complete.append(create_widgets(key, widget_info[0], row, col))

txtBox = tk.Text(root, height=25, width =110)
btn = tk.Button(root,text="Click Me", command=submit)
txtBox.grid(pady=10)
btn.grid()
root.mainloop()