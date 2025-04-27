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

url = "https://raw.githubusercontent.com/HarshiniAiyyer/Financial-Forensics/refs/heads/main/states.csv"

df = pd.read_csv(url)

df.isnull().sum()

df = df.dropna()
df = df.drop(columns = ['Uninsured Rate Change (2010-2015)'])

# Remove the percentages and dollar signs
def clean_percentage(value):
    if isinstance(value, str):
        if "%" in value:
            return value.replace('%', '')
        elif "$" in value:
            return value.replace('$', '').replace(',', '')
    return value

# Apply the cleaning function to all columns
df = df.map(clean_percentage)

# Loop through columns (excluding the first column) and convert 'object' columns to float
for col in df.columns[1:]:  # Exclude the first column by starting from index 1
    if df[col].dtype == 'object':  # Check if the column has 'object' type
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric (float), set errors to NaN if conversion fails


"""### ML algorithms Pipeline

#### Data Setup
"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


x = df.iloc[:,[3,4,5,6,7,9,10,11,12]].values

y = le.fit_transform(df.iloc[:,8])



from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# the scaler object (model)
scaler = StandardScaler()
# fit and transform the data
x = scaler.fit_transform(x)

"""#### Train and Test Split"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#turning to csv
pd.DataFrame(x_train).to_csv('C:\\Users\\harsh\\OneDrive\\Desktop\\fin\\x_train.csv')
pd.DataFrame(y_train).to_csv('C:\\Users\\harsh\\OneDrive\\Desktop\\fin\\y_train.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(accuracy_score(y_test, y_pred))


