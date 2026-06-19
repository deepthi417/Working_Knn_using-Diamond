# app.py

import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("diamond_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Diamond Price Prediction", page_icon="💎")

st.title("💎 Diamond Price Prediction")
st.write("Enter diamond characteristics to predict its price.")

# Numerical inputs
carat = st.number_input("Carat", min_value=0.0, value=0.50, step=0.01)
depth = st.number_input("Depth", min_value=0.0, value=61.5, step=0.1)
table = st.number_input("Table", min_value=0.0, value=55.0, step=0.1)
x = st.number_input("Length (x)", min_value=0.0, value=5.0, step=0.01)
y = st.number_input("Width (y)", min_value=0.0, value=5.0, step=0.01)
z = st.number_input("Depth (z)", min_value=0.0, value=3.0, step=0.01)

# Categorical inputs
cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.selectbox(
    "Color",
    ["D", "E", "F", "G", "H", "I", "J"]
)

clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

# Create dataframe
input_data = pd.DataFrame({
    "carat": [carat],
    "cut": [cut],
    "color": [color],
    "clarity": [clarity],
    "depth": [depth],
    "table": [table],
    "x": [x],
    "y": [y],
    "z": [z]
})

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Diamond Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
