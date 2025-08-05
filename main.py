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
