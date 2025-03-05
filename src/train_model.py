import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import bentoml

# Loading the prepared data
X_train = pd.read_csv('data/processed/X_train.csv').values
X_test = pd.read_csv('data/processed/X_test.csv').values
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model performance:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# Visualization of the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual values')
plt.ylabel('Predictions')
plt.title('Comparison of actual values and predictions')
plt.savefig('models/prediction_vs_actual.png')
plt.close()

# Saving the model with BentoML
bentoml.sklearn.save_model(
    "admission_model",
    model,
    signatures={
        "predict": {
            "batchable": True,
        }
    },
)

print("\nModel saved with BentoML under the name 'admission_model'")
