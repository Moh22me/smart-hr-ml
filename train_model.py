import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/hr_data.csv')

# 2. Data Preprocessing (Cleaning & Encoding)
print("Preprocessing data...")
# Drop columns that have no predictive value for Attrition
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df = df.drop(columns=columns_to_drop)

# Encode categorical features (Text to Numbers)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 3. Separate Features (X) and Target (y)
# 'Attrition' is what we want to predict (0 = No, 1 = Yes)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 4. Train/Test Split (80% training, 20% testing)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Classification Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# 6. Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("="*30)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("="*30)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save the model and encoders for the Streamlit Dashboard
print("\nSaving model to 'models/' directory...")
joblib.dump(model, 'models/rf_model.pkl')
joblib.dump(label_encoders, 'models/encoders.pkl')
joblib.dump(X.columns.tolist(), 'models/features.pkl')

print("Success! Model training is complete.")