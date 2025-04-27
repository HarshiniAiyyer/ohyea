import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from benfordslaw import benfordslaw
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import metrics

# Set page config
st.set_page_config(page_title="Financial Forensics Dashboard", layout="wide")

# App title
st.title("Financial Forensics Dashboard")
st.markdown("""
This app performs **financial forensics analysis** using **Benford's Law** and machine learning techniques.
The data is loaded directly from GitHub.
""")

# Explanation of Benford's Law
st.markdown("""
**Benford's Law** states that in naturally occurring numerical datasets, the **leading digit is more likely to be small (1 appears more often than 9).**
It is commonly used in **financial fraud detection**.
""")

# Function to clean percentage and dollar signs
def clean_numeric(value):
    if isinstance(value, str):
        return value.replace('%', '').replace('$', '').replace(',', '')
    return value

# GitHub data URL input
github_url = st.text_input("Enter GitHub CSV URL:", 
                           "https://raw.githubusercontent.com/HarshiniAiyyer/Financial-Forensics/refs/heads/main/states.csv")

# Function to load data from GitHub
@st.cache_data
def load_github_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        # Data Cleaning: Remove symbols and convert to numeric
        df = df.applymap(clean_numeric)
        for col in df.columns[1:]:  
            df[col] = pd.to_numeric(df[col], errors="coerce")  
        
        # Drop missing values
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        return None

# Load and display data
if st.button("Load Data from GitHub"):
    with st.spinner("Loading data..."):
        df = load_github_data(github_url)
        if df is not None:
            st.session_state.data = df
            st.subheader("Raw Data")
            st.dataframe(df.head())

# ====== BENFORD'S LAW ANALYSIS ======
st.subheader("Benford's Law Analysis")

if "data" in st.session_state:
    df = st.session_state.data
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if numeric_columns:
        benford_column = st.selectbox("Select a column for Benford's Law analysis:", numeric_columns)

        if st.button("Run Benford's Law Analysis"):
            x = np.round(df[benford_column].dropna().values).astype(int)

            tab1, tab2, tab3, tab4 = st.tabs(["First Digit", "Second Digit", "Last Digit", "Second Last Digit"])

            with tab1:
                bl1 = benfordslaw(alpha=0.05)
                bl1.fit(x)
                fig, _ = bl1.plot()
                st.pyplot(fig)

            with tab2:
                bl2 = benfordslaw(pos=2)
                bl2.fit(x)
                fig, _ = bl2.plot()
                st.pyplot(fig)

            with tab3:
                bllast = benfordslaw(pos=-1)
                bllast.fit(x)
                fig, _ = bllast.plot()
                st.pyplot(fig)

            with tab4:
                blseclast = benfordslaw(pos=-2)
                blseclast.fit(x)
                fig, _ = blseclast.plot()
                st.pyplot(fig)


# ====== MACHINE LEARNING ANALYSIS ======
st.subheader("Machine Learning Analysis")

if "data" in st.session_state:
    df = st.session_state.data
    
    # Predefined feature and target indices
    feature_indices = [3,4,5,6,7,9,10,11,12]  
    target_index = 9  

    # Extract column names based on indices
    feature_columns = [df.columns[i] for i in feature_indices]
    target_column = df.columns[target_index]

    # Display fixed selections
    st.write(f"**Features Selected:** {feature_columns}")
    st.write(f"**Target Selected:** {target_column}")

    # Extract data
    X = df[feature_columns].values  
    y = df[target_column].values

    # Label encode target if it's categorical
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df[target_column])
        st.write(f"Target classes: {list(le.classes_)}")

    # Test size selection
    test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)

    # Train Models Button
    if st.button("Train Models"):
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Machine Learning Models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(kernel='linear', probability=True),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'MLP Classifier': MLPClassifier(max_iter=300),
            'XGBoost': XGBClassifier(),
            'AdaBoost': AdaBoostClassifier()
        }

        results = []
        progress_bar = st.progress(0)

        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            results.append({'Model': name, 'Accuracy': accuracy})
            progress_bar.progress((i + 1) / len(models))

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Accuracy comparison
        st.write("### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Model", y="Accuracy", data=results_df)
        plt.xticks(rotation=90)
        st.pyplot(fig)
