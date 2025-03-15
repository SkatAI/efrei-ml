import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming 'data_normalized' is already created with X1, X2, and y

df = pd.read_csv('./../data/Non-linear_Regression_Dataset.csv')

# Features and target
X = df[['X1', 'X2']].copy()
y = df['y'].copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg = DecisionTreeRegressor(max_depth = 2)
# benchmark
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"-- Mean Squared Error: {mse:.4f}")
print(f"-- R² Score: {r2:.4f}")


# Initialize Linear Regression as base estimator
estimator = LinearRegression()
estimator = DecisionTreeRegressor(max_depth = 10)
# Create Bagging Regressor
bagging_model = BaggingRegressor(estimator=estimator, n_estimators=50, random_state=42)

# Train the Bagging Regressor
bagging_model.fit(X_train, y_train)

# Make predictions
y_pred = bagging_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
