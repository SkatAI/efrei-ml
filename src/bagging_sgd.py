from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Generate synthetic non-linear data
np.random.seed(42)
X = np.random.uniform(-10, 10, (1000, 2))
y = np.sin(X[:, 0]) + np.log(np.abs(X[:, 1]) + 1) + np.random.normal(0, 0.1, 1000)

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

df = pd.read_csv('./../data/Non-linear_Regression_Dataset.csv')

# Features and target
X = df[['X1', 'X2']].copy()
y = df['y'].copy()

# -------- benchmark
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize high-variance SGD Regressor
sgd = SGDRegressor(tol=None, max_iter=10000, fit_intercept=True, shuffle=False, learning_rate='constant', eta0=0.1, penalty = None)
sgd.fit(X_train, y_train)
print(f"Mean Squared Error train: {mean_squared_error(y_train, sgd.predict(X_train)):.4f}")
print(f"Mean Squared Error test: {mean_squared_error(y_test, sgd.predict(X_test)):.4f}")


print(f"R² Score: {r2:.4f}")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# -------- bagging
sgd2 = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.001)
# Bagging Regressor with SGD as base estimator
bagging_sgd = BaggingRegressor(estimator=sgd2, n_estimators=50, random_state=42)

# Train the Bagging model
bagging_sgd.fit(X_train, y_train)

# Predictions
y_pred = bagging_sgd.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
