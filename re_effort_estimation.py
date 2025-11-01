import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Requirement_Estimation_Dataset.csv")

print("First 5 rows of data:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Effort_Hours"])  
y = df["Effort_Hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.3f} hours")
print(f"R² Score: {r2:.3f}")

cv_rmse = np.sqrt(-cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)).mean()
print(f"Cross-Validated RMSE: {cv_rmse:.3f} hours")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance (Top Predictors):")
print(importance.head())

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Effort (hours)")
plt.ylabel("Predicted Effort (hours)")
plt.title("Actual vs Predicted Effort")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x='Coefficient', y='Feature')
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.tight_layout()
plt.show()