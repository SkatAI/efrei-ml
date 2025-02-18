import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate artificial data
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = X**3 - 10*X + 4*np.random.normal(0, 2, X.shape[0])
print(y.shape)

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define polynomial degrees to test
degrees = [1, 3, 8,  20]
plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees):
    plt.subplot(1, len(degrees), i+1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    # Predictions
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    y_pred = model.predict(X_test)

    # Plot results
    plt.scatter(X_train, y_train, color='blue', label='Train Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')
    plt.plot(X_plot, y_plot, color='black', linewidth=2, label=f'Poly Degree {degree}')
    plt.legend()
    plt.title(f'Degree {degree}\nMSE: {mean_squared_error(y_test, y_pred):.2f}')

plt.tight_layout()
plt.show()

for i, degree in enumerate(degrees):
    plt.subplot(1, len(degrees), i+1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    # Predictions
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    y_pred = model.predict(X_test)

    # Plot results
    plt.scatter(X_train, y_train, color='blue', label='Train Data')
    plt.scatter(X_test, y_test, color='red', label='Test Data')
    plt.plot(X_plot, y_plot, color='black', linewidth=2, label=f'Poly Degree {degree}')
    plt.legend()
    plt.title(f'Degree {degree}\nMSE: {mean_squared_error(y_test, y_pred):.2f}')

plt.tight_layout()
plt.show()
