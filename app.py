# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Page config
st.set_page_config(page_title="Madical Insurance Cost Explorer", layout="centered")
st.title("💡 Medical Insurance Cost Prediction App")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file (must contain 'charges')", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your dataset to begin.")
    st.stop()

# Load data
data = pd.read_csv(uploaded_file)

# Auto-detect cost column
guessed_targets = [c for c in data.columns if c.lower() == "charges"]

if guessed_targets:
    target_col = guessed_targets[0]
    st.success(f"Using '{target_col}' as the cost column.")
else:
    target_col = st.selectbox(
        "Select your target (cost/charges) column:",
        options=data.columns
    )

# Automatically detect features
all_feature_cols = [c for c in data.columns if c != target_col]
numeric_cols = [c for c in all_feature_cols if pd.api.types.is_numeric_dtype(data[c])]
categorical_cols = [c for c in all_feature_cols if c not in numeric_cols]

feature_cols = numeric_cols + categorical_cols

st.markdown("**Feature Columns:** " + ", ".join(feature_cols))

# Prepare data
X = data[feature_cols].copy()
y = data[target_col]

# One-hot encode categoricals
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Show metrics
st.subheader("📈 Model Evaluation")
st.metric("R² Score", f"{r2:.3f}")

# Prediction form
st.subheader("🧍 Predict Insurance Cost for a New Entry")

with st.form("predict_form"):
    input_data = {}

    for col in numeric_cols:
        if col.lower() == "bmi":
            val = st.number_input(
                f"{col}", 
                value=float(data[col].mean()), 
                step=0.1, 
                format="%.1f"
            )
            input_data[col] = float(val)
        else:
            val = st.number_input(
                f"{col}", 
                value=int(data[col].mean()), 
                step=1, 
                format="%d"
            )
            input_data[col] = int(val)

    for col in categorical_cols:
        options = data[col].unique().tolist()
        choice = st.selectbox(f"{col}", options=options)
        input_data[col] = choice

    submitted = st.form_submit_button("📊 Predict")
    if submitted:
        user_df = pd.DataFrame([input_data])
        user_df = pd.get_dummies(user_df, columns=categorical_cols, drop_first=True)
        user_df = user_df.reindex(columns=X_train.columns, fill_value=0)
        prediction = model.predict(user_df)[0]
        st.success(f"💰 Estimated Insurance Cost: $ {prediction:,.0f}")

st.caption("Model: scikit-learn Linear Regression | Author: You 🚀")
