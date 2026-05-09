# import pandas as pd
# import joblib

# # Load model
# model = joblib.load("house_price_model.pkl")

# # Load test data
# test_df = pd.read_csv("archive_3/test.csv")

# # Save Id separately
# ids = test_df["Id"]

# # Drop Id
# test_df = test_df.drop("Id", axis=1)

# # Predict
# predictions = model.predict(test_df)

# # Create submission file
# submission = pd.DataFrame({
#     "Id": ids,
#     "SalePrice": predictions
# })

# submission.to_csv("archive_3/submission.csv", index=False)

# print("Submission file created!")


import pandas as pd
import joblib

# =========================
# Load trained model
# =========================
model = joblib.load("house_price_model.pkl")

# =========================
# Load test data
# =========================
test_df = pd.read_csv("archive_3/test.csv")

# Save Id column
ids = test_df["Id"]

# Drop Id column
test_df = test_df.drop("Id", axis=1)

# =========================
# Handle missing values
# =========================
test_df = test_df.fillna(test_df.mean(numeric_only=True))

# =========================
# Make predictions
# =========================
predictions = model.predict(test_df)

# =========================
# Create submission file
# =========================
submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": predictions
})

submission.to_csv("archive_3/submission.csv", index=False)

print("✅ Submission file created successfully!")