import pandas as pd
import os
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Load Dataset
# -----------------------------

data_path = "data/teacher_data.csv"

df = pd.read_csv(data_path)


print("Dataset Loaded Successfully")
print("--------------------------------")

print(df.head())


# -----------------------------
# 2. Dataset Information
# -----------------------------

print("\nDataset Shape:")
print(df.shape)


print("\nColumn Information:")
print(df.info())


# -----------------------------
# 3. Check Missing Values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# 4. Remove Duplicates
# -----------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicate_count)


df = df.drop_duplicates()


# -----------------------------
# 5. Handle Missing Values
# -----------------------------

for column in df.columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mean()
        )


print("\nMissing values handled")


# -----------------------------
# 6. Separate Features and Target
# -----------------------------

X = df.drop(
    "Performance_Score",
    axis=1
)


y = df[
    "Performance_Score"
]


print("\nFeatures:")
print(X.head())


print("\nTarget:")
print(y.head())


# -----------------------------
# 7. Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data:")
print(X_train.shape)


print("\nTesting Data:")
print(X_test.shape)



# -----------------------------
# 8. Save Processed Data
# -----------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)


X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)


y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)


y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)


print("\nProcessed data saved successfully!")
