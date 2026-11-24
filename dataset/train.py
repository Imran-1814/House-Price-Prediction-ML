from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data_path = Path(__file__).resolve().parent / "House Price Prediction Dataset.csv"
df = pd.read_csv(data_path)

# 2. Drop 'Id' and clean duplicates/missing rows
df = df.drop(columns=["Id"]).drop_duplicates().dropna()

# 3. Check numeric correlations with Price
print("Correlation with Price:\n", df.corr(numeric_only=True)["Price"].sort_values(ascending=False))

# 4. Feature Engineering
df["HouseAge"] = 2026 - df["YearBuilt"]
df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]

# 5. Separate features (X) and target (y)
X = df.drop(columns=["Price"])
y = df["Price"]

# 6. One-hot encode categorical columns (Location, Condition, Garage)
X = pd.get_dummies(X, drop_first=True)

# 7. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Compare Linear Regression vs Random Forest
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_r2 = float("-inf")

print("\n--- Model Comparison ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"{name} -> RMSE: {rmse:,.2f} | R2 Score: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# 9. Save the best model and feature columns for future predictions
model_dir = Path(__file__).resolve().parent
joblib.dump({"model": best_model, "columns": X.columns.tolist()}, model_dir / "house_price_model.pkl")
print("\nSaved best model to house_price_model.pkl")

# 10. Test prediction on one sample from the test set
sample_house = X_test.iloc[[0]]
predicted_price = best_model.predict(sample_house)[0]
actual_price = y_test.iloc[0]
print(f"Sample Prediction: {predicted_price:,.2f} | Actual Price: {actual_price:,.2f}")