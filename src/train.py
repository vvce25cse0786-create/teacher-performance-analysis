import pandas as pd
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ====================================
# 1. Load Processed Data
# ====================================

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)


y_train = pd.read_csv(
    "data/processed/y_train.csv"
)


y_test = pd.read_csv(
    "data/processed/y_test.csv"
)


print("Data Loaded Successfully")


# ====================================
# 2. Convert Target Data
# ====================================

# Convert dataframe to series

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()



# ====================================
# 3. Create Machine Learning Model
# ====================================

model = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)



# ====================================
# 4. Train Model
# ====================================

print("Training Started...")

model.fit(
    X_train,
    y_train
)


print("Training Completed")



# ====================================
# 5. Prediction
# ====================================

y_pred = model.predict(
    X_test
)



# ====================================
# 6. Model Evaluation
# ====================================


mae = mean_absolute_error(
    y_test,
    y_pred
)


mse = mean_squared_error(
    y_test,
    y_pred
)


rmse = mse ** 0.5


r2 = r2_score(
    y_test,
    y_pred
)



print("\nModel Performance")
print("-------------------------")

print("MAE :", mae)

print("RMSE :", rmse)

print("R2 Score :", r2)



# ====================================
# 7. Save Model
# ====================================


os.makedirs(
    "models",
    exist_ok=True
)


model_path = "models/model.pkl"


joblib.dump(
    model,
    model_path
)


print("\nModel Saved Successfully")

print(
    "Location:",
    model_path
)
