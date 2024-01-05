import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Load Iris dataset
iris = load_iris()

# Extract target and feature names
target_names = iris.target_names
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=feature_names)

# Add target column
y = [target_names[target] for target in iris.target]
df['target'] = y

# Split into features (X) and target labels (y)
X = df.iloc[:, :-1]
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Algorithms
algorithms = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Streamlit setup
st.set_page_config(
    page_title="Iris Classification",
    page_icon=":sunflower:"
)

# Streamlit components
st.title("Iris Classification")
st.write("Choose an algorithm and input sepal/petal dimensions for multi-prediction.")

# User input for sepal and petal dimensions
sepal_length = st.number_input(label="Sepal Length", min_value=df['sepal_length'].min(), max_value=df['sepal_length'].max())
sepal_width = st.number_input(label="Sepal Width", min_value=df['sepal_width'].min(), max_value=df['sepal_width'].max())
petal_length = st.number_input(label="Petal Length", min_value=df['petal_length'].min(), max_value=df['petal_length'].max())
petal_width = st.number_input(label="Petal Width", min_value=df['petal_width'].min(), max_value=df['petal_width'].max())

# Dropdown for algorithm selection
selected_algorithm = st.selectbox("Select Algorithm", list(algorithms.keys()))

# Button for prediction
predict_btn = st.button("Predict", type="primary")

# Perform prediction on button click
prediction = ":violet[-]"
if predict_btn:
    model = algorithms[selected_algorithm]
    inputs = [[sepal_length, sepal_width, petal_length, petal_length]]
    prediction = model.fit(X_train, y_train).predict(inputs)[0]

# Display the prediction result
st.write("")
st.write("")
st.subheader("Prediction:")
st.subheader(prediction)
