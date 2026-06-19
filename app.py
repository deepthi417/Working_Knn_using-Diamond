# app.py

import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# -------------------------
# Load Model and Preprocessor
# -------------------------
@st.cache_resource
def load_objects():
    with open("diamond_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_objects()

# -------------------------
# Title
# -------------------------
st.title("💎 Diamond Price Prediction")
st.markdown("Enter the diamond characteristics below and click **Predict Price**.")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Sample Values")

st.sidebar.info("""
Carat : 0.75
Cut : Premium
Color : G
Clarity : VS2
Depth : 61.7
Table : 58
x : 5.79
y : 5.82
z : 3.58
""")

# -------------------------
# Input Fields
# -------------------------
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.0, value=0.75, step=0.01)
    depth = st.number_input("Depth", min_value=0.0, value=61.7, step=0.1)
    table = st.number_input("Table", min_value=0.0, value=58.0, step=0.1)

with col2:
    x = st.number_input("Length (x)", min_value=0.0, value=5.79, step=0.01)
    y = st.number_input("Width (y)", min_value=0.0, value=5.82, step=0.01)
    z = st.number_input("Depth (z)", min_value=0.0, value=3.58, step=0.01)

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

# -------------------------
# Create Input DataFrame
# -------------------------
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

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price 💎"):

    try:
        # Transform input
        transformed_input = preprocessor.transform(input_data)

        transformed_input = pd.DataFrame(
            transformed_input,
            columns=preprocessor.get_feature_names_out()
        )

        # Predict
        prediction = model.predict(transformed_input)[0]

        st.success(
            f"💰 Estimated Diamond Price: ${prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Diamond Price Prediction using KNN Regression")
