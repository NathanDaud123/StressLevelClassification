import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import requests

# Load the dataset
url = 'https://docs.google.com/uc?export=download&id=184-NNnzhNmB1IIjz-7YTW5SrkVFT6fF_'
response = requests.get(url)
with open('StressLevelDataset.csv', 'w') as file:
    file.write(response.text)
data = pd.read_csv("StressLevelDataset.csv")

# Split the dataset
X = data.drop('stress_level', axis=1)
y = data['stress_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create classifiers
weak_classifier = DecisionTreeClassifier(max_depth=1)
ada_classifier = AdaBoostClassifier(weak_classifier, n_estimators=50, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit classifiers to the training data
ada_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)

# Streamlit app
st.title("Stress Level Prediction")

# Add a section to display dataset and user input
st.header("Dataset Overview:")
st.write(data.head())

st.header("Input Your Characteristic")
# Collect user input
user_input_dict = {}
for column in data.columns[:-1]:
    min_value = data[column].min()
    max_value = data[column].max()

    user_input = st.number_input(
        label=f"Enter {column} value ({min_value} - {max_value})",
        min_value=min_value,
        max_value=max_value,
        value=min_value
    )

    user_input_dict[column] = user_input

# Button to trigger classification
if st.button("Predict Stress Level"):
    user_input_data = pd.DataFrame([user_input_dict])
    
    # Predict with AdaBoost
    ada_prediction = ada_classifier.predict(user_input_data)
    # Predict with Random Forest
    rf_prediction = rf_classifier.predict(user_input_data)

    # Calculate and display accuracy
    st.subheader("Prediction Results:")
    st.write(f"AdaBoost Prediction: {ada_prediction[0]}")
    st.write(f"Random Forest Prediction: {rf_prediction[0]}")

    # Evaluate accuracy
    y_pred_ada = ada_classifier.predict(X_test)
    y_pred_rf = rf_classifier.predict(X_test)

    accuracy_ada = accuracy_score(y_test, y_pred_ada)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    st.subheader("Model Accuracy:")
    st.write(f"AdaBoost Accuracy: {accuracy_ada:.2%}")
    st.write(f"Random Forest Accuracy: {accuracy_rf:.2%}")

# Add a sidebar with dataset info and explanation
st.sidebar.title("Stress Level Prediction")
st.sidebar.write("Stress level prediction is a machine learning task where the goal is to predict the stress level of an individual based on certain characteristics. In this app, we use two algorithms, AdaBoost and Random Forest, to make predictions.")

st.sidebar.subheader("AdaBoost Algorithm:")
st.sidebar.write("AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak classifiers to create a strong classifier. In this app, a decision tree with a maximum depth of 1 is used as the weak classifier.")

st.sidebar.subheader("Random Forest Algorithm:")
st.sidebar.write("Random Forest is an ensemble learning method that builds a multitude of decision trees and merges them together to get a more accurate and stable prediction. In this app, a Random Forest classifier with 100 trees is used.")

st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"Number of Samples: {len(data)}")
st.sidebar.write(f"Number of Features: {len(data.columns) - 1}")
st.sidebar.write(f"Target Variable: stress_level")
