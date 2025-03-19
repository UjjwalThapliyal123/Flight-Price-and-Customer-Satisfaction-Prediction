import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import re
from streamlit_echarts import st_echarts

stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}

# Function to convert Duration to minutes
def convert_duration(duration):
    duration = str(duration)  # Ensure string format
    hours = re.findall(r'(\d+)h', duration)
    minutes = re.findall(r'(\d+)m', duration)
    total_minutes = int(hours[0]) * 60 if hours else 0
    total_minutes += int(minutes[0]) if minutes else 0
    return total_minutes

# Load Data Function
@st.cache_data
def load_data(file_name):
    try:
        df = pd.read_csv(file_name, low_memory=False)
        return df
    except FileNotFoundError:
        st.error(f"Error: {file_name} not found. Please upload the dataset.")
        st.stop()
        
        
st.title("Flight Price and Passenger Satisfaction prediction")

page = st.sidebar.radio("Go to Model", ["Regression Models", "Classification Models","Flight Price Prediction", "Customer Satisfaction Prediction","Feedback"])

if page == "Regression Models":
        
    st.write(
        """
        This Streamlit app provides an interactive explanation of three popular machine learning models: 
        **Linear Regression, Random Forest, and XGBoost.**  
        Select a model to learn more!  
        """
    )

    # User selects a model to explore
    model_option = st.selectbox(
        "🔍 Select a Model:",
        ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]
    )

    # Generate Sample Data for Visualization
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = 2 * X.flatten() + 3 + np.random.normal(0, 2, 100)  # True line with noise

    # ---------------------- LINEAR REGRESSION ----------------------
    if model_option == "Linear Regression":
        st.subheader("📊 Linear Regression")
        st.write(
            "- **Type:** Simple, interpretable regression model.\n"
            "- **Best for:** Predicting continuous values (e.g., house prices, sales, stock prices).\n"
            "- **How it works:** Finds a straight-line relationship between input (X) and output (Y).\n"
            "- **Pros:** Easy to understand, fast to train.\n"
            "- **Cons:** Performs poorly on **non-linear** data."
        )

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X, y_true)
        y_pred = model.predict(X)

        # Plot Graph
        fig, ax = plt.subplots()
        ax.scatter(X, y_true, label="Actual Data", color="blue")
        ax.plot(X, y_pred, color="red", label="Linear Regression Line")
        ax.set_title("Linear Regression Fit")
        ax.legend()
        st.pyplot(fig)

    # ---------------------- RANDOM FOREST REGRESSOR ----------------------
    elif model_option == "Random Forest Regressor":
        st.subheader("🌳 Random Forest Regressor")
        st.write(
            "- **Type:** Ensemble learning model using multiple decision trees.\n"
            "- **Best for:** Handling complex relationships and noisy data.\n"
            "- **How it works:** Combines predictions from multiple trees to improve accuracy.\n"
            "- **Pros:** Reduces overfitting, works well with missing data.\n"
            "- **Cons:** Slower than simple models, less interpretable."
        )

        # Train Random Forest Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y_true)
        y_pred = model.predict(X)

        # Plot Graph
        fig, ax = plt.subplots()
        ax.scatter(X, y_true, label="Actual Data", color="blue")
        ax.plot(X, y_pred, color="green", label="Random Forest Prediction", linewidth=2)
        ax.set_title("Random Forest Regression Fit")
        ax.legend()
        st.pyplot(fig)

    # ---------------------- XGBOOST REGRESSOR ----------------------
    elif model_option == "XGBoost Regressor":
        st.subheader("🚀 XGBoost Regressor")
        st.write(
            "- **Type:** Boosting algorithm for high-performance regression.\n"
            "- **Best for:** Large datasets with complex patterns.\n"
            "- **How it works:** Trains multiple weak models sequentially to minimize errors.\n"
            "- **Pros:** High accuracy, works well with outliers.\n"
            "- **Cons:** Computationally expensive, requires tuning."
        )

        # Train XGBoost Model
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y_true)
        y_pred = model.predict(X)

        # Plot Graph
        fig, ax = plt.subplots()
        ax.scatter(X, y_true, label="Actual Data", color="blue")
        ax.plot(X, y_pred, color="orange", label="XGBoost Prediction", linewidth=2)
        ax.set_title("XGBoost Regression Fit")
        ax.legend()
        st.pyplot(fig)

    st.success("🎯 Select another model to compare their performance!")
                
# ---------------------------- Classification Models Page ----------------------------
elif page == "Classification Models":
    def classification_intro():
            st.header(" Customer Satisfaction Prediction Models")
            st.write("Classification is used to predict customer satisfaction...")

            st.write(
                "Classification is a type of supervised learning where the goal is to predict categories (e.g., **Satisfied vs. Not Satisfied**). "
                "In this app, we use three powerful classifiers: **Logistic Regression, Random Forest, and Gradient Boosting**."
            )

            model_option = st.selectbox(
                "🔍 Select a Model to Learn More:", 
                ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"]
            )

            if model_option == "Logistic Regression":
                st.subheader("📊 Logistic Regression")
                st.write(
                    "- **Type:** Linear Model\n"
                    "- **Best for:** Binary classification problems\n"
                    "- **How it works:** Uses a sigmoid function to estimate probabilities\n"
                    "- **Pros:** Simple, fast, interpretable\n"
                    "- **Cons:** Works best only for **linearly separable** data"
                )
                st.image("https://tse4.mm.bing.net/th?id=OIP.v2W8CMoKDgVPPje0fXpItwAAAA&pid=Api&P=0&h=180", 
                        caption="Sigmoid Curve in Logistic Regression")

            elif model_option == "Random Forest Classifier":
                st.subheader("🌲 Random Forest Classifier")
                st.write(
                    "- **Type:** Ensemble Learning (Multiple Decision Trees)\n"
                    "- **Best for:** Handling non-linearity and noisy data\n"
                    "- **How it works:** Builds multiple decision trees and combines results\n"
                    "- **Pros:** Reduces overfitting, handles missing values well\n"
                    "- **Cons:** Slower for large datasets"
                )
                st.image("https://www.researchgate.net/publication/358926904/figure/download/fig1/AS:1128819838713974@1646143030852/Flowchart-of-random-forest-algorithm.jpg", 
                        caption="How Random Forest Works")

            elif model_option == "Gradient Boosting Classifier":
                st.subheader("🚀 Gradient Boosting Classifier")
                st.write(
                    "- **Type:** Boosting Algorithm (Sequential Learning)\n"
                    "- **Best for:** Complex datasets with non-linear patterns\n"
                    "- **How it works:** Improves weak learners by focusing on misclassified examples\n"
                    "- **Pros:** High accuracy, good for imbalanced data\n"
                    "- **Cons:** Computationally expensive, sensitive to overfitting"
                )
                st.image("https://datascience.eu/wp-content/uploads/2020/08/482246_1_En_25_Fig2_HTML-978x652.png", 
                        caption="Gradient Boosting ")

            st.success("🎯 Choose a model and explore its power in classification!")

# Run the function 
    if __name__ == "__main__":
        classification_intro()
        
# ---------------------- Flight Price Prediction ----------------------
if page == "Flight Price Prediction":
    st.header("📈 Flight Price Prediction")

    df_price = load_data("Flight_Price.csv")

    features = ['Airline', 'Source', 'Destination', 'Duration', 'Total_Stops', 'Additional_Info', 'Route']
    target = 'Price'

    if target in df_price.columns:
        X = df_price[features].copy()
        y = df_price[target]

        # Convert 'Duration' to minutes
        X['Duration'] = X['Duration'].apply(convert_duration)

        # Map 'Total_Stops' to numerical values
        X['Total_Stops'] = X['Total_Stops'].map(stops_mapping)

        # One-Hot Encoding for categorical features
        X = pd.get_dummies(X, columns=['Airline', 'Source', 'Destination', 'Additional_Info', 'Route'], drop_first=True)

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy='most_frequent')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standard Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Sidebar Model Selection
        model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost"], key="reg_model")

        # Model Selection
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        else:
            model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

        # Train Model
        model.fit(X_train, y_train)

        # Sidebar User Input
        st.sidebar.header("Enter Flight Details")
        selected_airline = st.sidebar.selectbox("Airline", df_price['Airline'].unique())
        selected_source = st.sidebar.selectbox("Source", df_price['Source'].unique())
        selected_destination = st.sidebar.selectbox("Destination", df_price['Destination'].unique())
        duration = st.sidebar.slider("Duration (hours)", min_value=0, max_value=24, value=2)
        total_stops = st.sidebar.radio("Stops", list(stops_mapping.keys()), key="stops_radio")
        selected_additional_info = st.sidebar.selectbox("Additional Info", df_price['Additional_Info'].unique())
        selected_route = st.sidebar.selectbox("Route", df_price['Route'].unique())

        # Convert Input Data
        input_data = pd.DataFrame({
            'Airline': [selected_airline],
            'Source': [selected_source],
            'Destination': [selected_destination],
            'Duration': [duration * 60],
            'Total_Stops': [stops_mapping.get(total_stops, -1)],
            'Additional_Info': [selected_additional_info],
            'Route': [selected_route]
        })

        # Apply One-Hot Encoding and Imputer to Input Data
        input_data = pd.get_dummies(input_data, columns=['Airline', 'Source', 'Destination', 'Additional_Info', 'Route'], drop_first=True)

        # Add missing columns
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Match Column Order
        input_data = input_data[X.columns]
        input_data = scaler.transform(input_data)

        # Predict Price
        if st.sidebar.button("Predict Flight Price"):
            prediction = model.predict(input_data)
            st.metric(label="Predicted Flight Price", value=f"₹{prediction[0]:,.2f}")

            
# Customer Satisfaction Prediction
elif page == "Customer Satisfaction Prediction":

    df_satisfaction = load_data("Passenger_Satisfaction.csv")  

    # Remove unnamed columns and drop 'User ID' 
    df_satisfaction = df_satisfaction.loc[:, ~df_satisfaction.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    if 'User ID' in df_satisfaction.columns:
        df_satisfaction = df_satisfaction.drop(columns=['User ID'])  # Remove User ID if it exists

    df_satisfaction = df_satisfaction.dropna()

    # Convert categorical columns to numeric
    def preprocess_data(df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes  # Encode categorical columns as numbers
        return df

    df_satisfaction = preprocess_data(df_satisfaction)

    target = 'satisfaction'
    features = [col for col in df_satisfaction.columns if col != target]
    X = df_satisfaction[features]
    y = df_satisfaction[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cache model training to speed up performance
    @st.cache_resource
    def train_model(model_type):
        if model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    model, accuracy = train_model(model_choice)

    # Sidebar User Input
    st.sidebar.header("Enter Passenger Details")
    input_data = {}
    for col in features:
        min_val, max_val, mean_val = float(X[col].min()), float(X[col].max()), float(X[col].mean())
        input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, step=(max_val - min_val) / 100)

    input_df = pd.DataFrame([input_data])

    # Prediction and Model Accuracy
    if st.sidebar.button("Predict Satisfaction"):
        prediction = model.predict(input_df)
        result_text = "Satisfied" if prediction[0] == 1 else "Not Satisfied"

        st.write(f"###  Predicted Satisfaction: **{result_text}**")
        st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
        
  # ---------------------------- Feedback Page ----------------------------
elif page == "Feedback":
    st.subheader("Give Feedback")
    text = st.text_area("Please provide your feedback here")
    rating = st.slider("Rating (1-5)", 1, 5, 3)
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")     
        st.balloons()
