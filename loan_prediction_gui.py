import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('loan_pred_status')

# Function to make prediction
def make_prediction():
    try:
        data = pd.DataFrame({
            'Gender': [int(gender_var.get())],
            'Married': [int(married_var.get())],
            'Dependents': [int(dependents_var.get())],
            'Education': [int(education_var.get())],
            'Self_Employed': [int(self_employed_var.get())],
            'ApplicantIncome': [int(applicant_income_var.get())],
            'CoapplicantIncome': [int(coapplicant_income_var.get())],
            'LoanAmount': [int(loan_amount_var.get())],
            'Loan_Amount_Term': [int(loan_amount_term_var.get())],
            'Credit_History': [int(credit_history_var.get())],
            'Property_Area': [int(property_area_var.get())]
        })

        result = model.predict(data)

        if result[0] == 1:
            messagebox.showinfo("Prediction Result", "Loan Approved")
        else:
            messagebox.showinfo("Prediction Result", "Loan Not Approved")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Create the main window
root = tk.Tk()
root.title("Loan Prediction GUI")

# Define the variables
gender_var = tk.StringVar()
married_var = tk.StringVar()
dependents_var = tk.StringVar()
education_var = tk.StringVar()
self_employed_var = tk.StringVar()
applicant_income_var = tk.StringVar()
coapplicant_income_var = tk.StringVar()
loan_amount_var = tk.StringVar()
loan_amount_term_var = tk.StringVar()
credit_history_var = tk.StringVar()
property_area_var = tk.StringVar()

# Create the input fields
tk.Label(root, text="Gender (1=Male, 0=Female)").grid(row=0)
tk.Entry(root, textvariable=gender_var).grid(row=0, column=1)
tk.Label(root, text="Married (1=Yes, 0=No)").grid(row=1)
tk.Entry(root, textvariable=married_var).grid(row=1, column=1)
tk.Label(root, text="Dependents").grid(row=2)
tk.Entry(root, textvariable=dependents_var).grid(row=2, column=1)
tk.Label(root, text="Education (1=Graduate, 0=Not Graduate)").grid(row=3)
tk.Entry(root, textvariable=education_var).grid(row=3, column=1)
tk.Label(root, text="Self Employed (1=Yes, 0=No)").grid(row=4)
tk.Entry(root, textvariable=self_employed_var).grid(row=4, column=1)
tk.Label(root, text="Applicant Income").grid(row=5)
tk.Entry(root, textvariable=applicant_income_var).grid(row=5, column=1)
tk.Label(root, text="Coapplicant Income").grid(row=6)
tk.Entry(root, textvariable=coapplicant_income_var).grid(row=6, column=1)
tk.Label(root, text="Loan Amount").grid(row=7)
tk.Entry(root, textvariable=loan_amount_var).grid(row=7, column=1)
tk.Label(root, text="Loan Amount Term").grid(row=8)
tk.Entry(root, textvariable=loan_amount_term_var).grid(row=8, column=1)
tk.Label(root, text="Credit History (1=Yes, 0=No)").grid(row=9)
tk.Entry(root, textvariable=credit_history_var).grid(row=9, column=1)
tk.Label(root, text="Property Area (2=Urban, 1=Semiurban, 0=Rural)").grid(row=10)
tk.Entry(root, textvariable=property_area_var).grid(row=10, column=1)

# Create the prediction button
tk.Button(root, text="Predict", command=make_prediction).grid(row=11, columnspan=2)

# Run the application
root.mainloop()
