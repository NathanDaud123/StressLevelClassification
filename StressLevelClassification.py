from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

label_feature = data.columns[:]
dataset = data.columns.str.replace('_', ' ').str.title()

# Sidebar menu to select features
st.sidebar.header("Navigation")
navigation_menu = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Clustering", "Prediction"])

if navigation_menu == "Exploratory Data Analysis":
    st.title("Dataset Information")

    st.header("Dataset Kaggle")
    st.write(data)

    col = 3
    row = np.ceil((data.shape[1] - 1) / col).astype(int)
    fig, ax = plt.subplots(row, col, figsize=(14, 30))
    axi = ax.ravel()
    for i, c in enumerate(label_feature):
        uni = len(data[c].unique())
        axi[i].hist(data[c].tolist(), bins=uni, edgecolor='black')
        axi[i].set_title(c)
        axi[i].set_ylabel('Freq')
        if uni < 10:
            axi[i].set_xticks(data[c].unique())

    st.header("Visualization of Dataset")
    st.pyplot(fig)

    st.header("Corelation of Dataset")
    # Create a heatmap using Seaborn
    heat = plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(numeric_only=True), cmap='turbo')
    plt.title("Correlation Heatmap")

    # Display the heatmap using st.pyplot()
    st.pyplot(heat)

elif navigation_menu == "Clustering":
    st.sidebar.header("Select The Features")
    selected_feature1 = st.sidebar.selectbox('Select Feature 1', label_feature)
    selected_feature2 = st.sidebar.selectbox('Select Feature 2', label_feature)
    st.title("Clustering using KMeans")
    kmeans = KMeans(n_clusters=3)  # Set the number of clusters as needed
    data['Cluster'] = kmeans.fit_predict(data[label_feature])

    # Scatter plot for clustering
    clust = plt.figure(figsize=(8, 6))
    plt.scatter(data[selected_feature1], data[selected_feature2], c=data['Cluster'], cmap='viridis')
    plt.title('Clustering of Data Points')
    plt.xlabel(selected_feature1)
    plt.ylabel(selected_feature2)

    # Display the scatter plot using st.pyplot()
    st.pyplot(clust)

elif navigation_menu == "Prediction":
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
