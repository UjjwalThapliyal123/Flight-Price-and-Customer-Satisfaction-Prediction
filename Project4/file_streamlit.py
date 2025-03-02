import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ---------------------------- Load Dataset Function ----------------------------
@st.cache_data
def load_data_reg():
    df = pd.read_csv("flight_upd.csv")
    df = df.select_dtypes(include=[np.number])  
    df.dropna(inplace=True)
    return df.sample(frac=0.5, random_state=42)  # Reduce dataset size

# ---------------------------- Sidebar Navigation ----------------------------
st.sidebar.title("Navigate")    
page = st.sidebar.radio("Go to Model", ["Regression Models", "Classification Models", "Feedback"])
df = load_data_reg()

# ---------------------------- Regression Models Page ----------------------------
if page == "Regression Models":
    st.title("📊 Flight Price Prediction Models")
    st.image("fligh.jpg", caption="Flight Price Prediction", use_column_width=True)

    # Sidebar Configuration
    st.sidebar.header("Model Configuration")
    model_choice = st.sidebar.selectbox("Choose a Model", ["Linear Regression", "Random Forest Regression", "XGBoost"])

    # Data Preprocessing
    if "Price" in df.columns:
        X = df.drop(columns=["Price"])
        y = df["Price"]
    else:
        st.error("Dataset does not contain 'Price' column.")
        st.stop()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest Regression":
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)

    # Train & Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display Metrics
    st.sidebar.markdown("### Model Performance Metrics")
    st.sidebar.write(f"**RMSE:** {rmse:.2f}")
    st.sidebar.write(f"**R² Score:** {r2:.2f}")
    st.sidebar.write(f"**MAE:** {mae:.2f}")

# ---------------------------- Classification Models Page ----------------------------
elif page == "Classification Models":
    st.title("📊 Customer Satisfaction Prediction Models")
    st.image("pass.jpg", caption="Passenger Satisfaction", use_column_width=True)

    @st.cache_data
    def load_data_clas():
        passenger = pd.read_csv("PASS_SATISFICATION.csv")
        passenger.dropna(inplace=True)
        return passenger.sample(frac=0.5, random_state=42)

    passenger = load_data_clas()

    # Sidebar Configuration
    st.sidebar.header("Model Configuration")
    model_pick = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"])

    # Data Preprocessing
    if "satisfaction_satisfied" in passenger.columns:
        X = passenger.drop(columns=["satisfaction_satisfied"])
        y = passenger["satisfaction_satisfied"]
    else:
        st.error("Dataset does not contain 'satisfaction_satisfied' column.")
        st.stop()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    # Model Selection
    if model_pick == "Logistic Regression":
        model = LogisticRegression()
    elif model_pick == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=50)
    elif model_pick == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)

    # Train & Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display Metrics
    st.sidebar.markdown("### Model Performance Metrics")
    st.sidebar.write(f"Accuracy: {accuracy:.2f}")
    st.sidebar.write(f"Precision: {precision:.2f}")
    st.sidebar.write(f"Recall: {recall:.2f}")
    st.sidebar.write(f"F1-score: {f1:.2f}")

# ---------------------------- Feedback Page ----------------------------
elif page == "Feedback":
    st.subheader("Give Feedback")
    text = st.text_area("Please provide your feedback here")
    rating = st.slider("Rating (1-5)", 1, 5, 3)
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
