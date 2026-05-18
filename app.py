import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
# Make sure the CSV file name matches exactly
# Example: StudentsPerformance.csv

df = pd.read_csv('StudentsPerformance.csv')

# Convert categorical data

df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('math score', axis=1)
y = df['math score']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Frontend UI
st.title('Student Performance Prediction')
st.write('Predict student math score using Machine Learning')

# Input fields
reading_score = st.slider('Reading Score', 0, 100, 50)
writing_score = st.slider('Writing Score', 0, 100, 50)

# Create input dataframe
input_data = pd.DataFrame({
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# Add missing columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[X.columns]

# Prediction button
if st.button('Predict Math Score'):
    prediction = rf.predict(input_data)
    st.success(f'Predicted Math Score: {prediction[0]:.2f}')