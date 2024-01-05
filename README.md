# Iris Classification Web App

This is a simple web application for Iris flower classification. Users can choose between two classification algorithms (Decision Tree and K-Nearest Neighbors) to predict the species of an Iris flower based on its sepal and petal dimensions.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [License](#license)

## Introduction

This web app utilizes the Streamlit library to create a user interface for predicting the species of an Iris flower. The classification is performed using two algorithms: Decision Tree and K-Nearest Neighbors.

## Prerequisites

Make sure you have the following installed:

- Python (>=3.6)
- Pip (Python package installer)

## Installation

1. Change into the project directory:
   cd iris-classification-web-app

2. Install the required packages:
   pip install -r requirements.txt

3. Usage
   Run the web app using the following command:
   streamlit run app.py

Visit http://localhost:8501 in your web browser to use the application.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/iris-classification-web-app.git
   ```

Code Explanation
Import Libraries: The necessary libraries are imported, including Pandas, scikit-learn, and Streamlit.

Load Iris Dataset: The Iris dataset is loaded using the load_iris() function.

Extract Target and Feature Names: Target and feature names are extracted from the dataset.

Create DataFrame: A Pandas DataFrame is created using the Iris data.

Add Target Column: The target column is added to the DataFrame.

Split into Features and Target Labels: The dataset is split into features (X) and target labels (y).

Split into Training and Testing Sets: The data is split into training and testing sets using train_test_split.

Algorithms: Two classification algorithms, Decision Tree and K-Nearest Neighbors, are set up.

Streamlit Setup: Streamlit configuration with page title and icon.

Streamlit Components: Title and description are displayed.

User Input for Sepal and Petal Dimensions: User input is collected for sepal and petal dimensions.

Dropdown for Algorithm Selection: Users can choose the classification algorithm.

Button for Prediction: A button triggers the prediction.

Perform Prediction on Button Click: Prediction is performed when the button is clicked.

Display the Prediction Result: The prediction result is displayed on the web app.
