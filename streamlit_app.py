import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Design the Website UI Layout
st.title("Credit Risk Assessment App")

# Define the input features for the model
feature_names = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
    'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
    'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

# Define variables that need numeric input
scale_vars = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
              'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# Collect user inputs

user_inputs = {}
for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        # Collects a number for variables that need scaling
        user_inputs[feature] = st.number_input(f"Enter value for {feature}")
    else:
        # Collects text for other variables
        user_inputs[feature] = st.text_input(f"Enter value for {feature}")

# Wrap the dictionary in a list [] to make it a 2D sample
input_df = pd.DataFrame([user_inputs])

input_df['Education'] = pd.factorize(input_df['Education'])[0]
input_df['EmploymentType'] = pd.factorize(input_df['EmploymentType'])[0]
input_df['MaritalStatus'] = pd.factorize(input_df['MaritalStatus'])[0]
input_df['HasMortgage'] = pd.factorize(input_df['HasMortgage'])[0]
input_df['HasDependents'] = pd.factorize(input_df['HasDependents'])[0]
input_df['LoanPurpose'] = pd.factorize(input_df['LoanPurpose'])[0]
input_df['HasCoSigner'] = pd.factorize(input_df['HasCoSigner'])[0]

print("Expected features:", model.booster_.feature_name())
print("Provided features:", input_df.columns.tolist())

# Trigger Predictions with a Button
if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(input_df)

    # Display the final output
    st.success(f"The predicted value is: {prediction[0]}")
