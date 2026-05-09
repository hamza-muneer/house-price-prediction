# ================================
# HOUSE PRICE PREDICTION (AMES DATASET)
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ================================
# 1. Load Dataset
# ================================

df = pd.read_csv("archive_3/train.csv")

print("Dataset Loaded")
print(df.head())

# ================================
# 2. Separate Features & Target
# ================================

y = df["SalePrice"]   # target
X = df.drop(["SalePrice"], axis=1)

# Drop ID (not useful)
if "Id" in X.columns:
    X = X.drop("Id", axis=1)

# ================================
# 3. Handle Missing Values
# ================================

# Fill numeric columns with median
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Fill categorical columns with mode
cat_cols = X.select_dtypes(include=["object", "string"]).columns
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

# ================================
# 4. Preprocessing
# ================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ================================
# 5. Model Pipeline
# ================================

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

# ================================
# 6. Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 7. Train Model
# ================================

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ================================
# 8. Evaluate
# ================================

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2)

# ================================
# 9. Save Model
# ================================

joblib.dump(model, "house_price_model.pkl")
print("\nModel saved successfully")