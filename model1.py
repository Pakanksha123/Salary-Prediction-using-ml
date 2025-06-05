import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')

# Features and target
X = dataset[['YearsExperience', 'PreviousSalary']].values  # shape (n_samples, 2)
y = dataset['Salary'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Transform features into polynomial features (degree 2 example)
poly = PolynomialFeatures(degree=2, include_bias=False)  # degree can be changed

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Linear Regression on polynomial features
regressor = LinearRegression()
regressor.fit(X_train_poly, y_train)

# Predict on test set (optional)
y_pred = regressor.predict(X_test_poly)

# Save model and polynomial transformer
with open('model_poly.pkl', 'wb') as f:
    pickle.dump((regressor, poly), f)

# Load model and transformer
with open('model_poly.pkl', 'rb') as f:
    model, poly_transformer = pickle.load(f)

# Predict salary for new input: 1.8 years experience, previous salary 40000
input_features = np.array([[1.8, 40000]])
input_poly = poly_transformer.transform(input_features)
predicted_salary = model.predict(input_poly)

print(f"Predicted Salary: ${predicted_salary[0]:.2f}")
