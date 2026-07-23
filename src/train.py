import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================
# 1. Load Processed Data
# =====================================

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


# Convert dataframe to array

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()


print("Data Loaded Successfully")


# =====================================
# 2. MLflow Configuration
# =====================================

mlflow.set_experiment(
    "Teacher_Performance_Prediction"
)


# Start MLflow Run

with mlflow.start_run():


    # =================================
    # 3. Model Parameters
    # =================================

    n_estimators = 100

    random_state = 42



    # Log parameters

    mlflow.log_param(
        "model",
        "Random Forest Regressor"
    )


    mlflow.log_param(
        "n_estimators",
        n_estimators
    )


    mlflow.log_param(
        "random_state",
        random_state
    )



    # =================================
    # 4. Train Model
    # =================================

    model = RandomForestRegressor(

        n_estimators=n_estimators,

        random_state=random_state

    )


    model.fit(
        X_train,
        y_train
    )


    print("Model Training Completed")



    # =================================
    # 5. Prediction
    # =================================

    y_pred = model.predict(
        X_test
    )



    # =================================
    # 6. Evaluation Metrics
    # =================================


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



    print("-------------------------")
    print("MAE :", mae)
    print("RMSE :", rmse)
    print("R2 Score :", r2)



    # =================================
    # 7. Log Metrics in MLflow
    # =================================


    mlflow.log_metric(
        "MAE",
        mae
    )


    mlflow.log_metric(
        "RMSE",
        rmse
    )


    mlflow.log_metric(
        "R2_score",
        r2
    )



    # =================================
    # 8. Save Model
    # =================================


    os.makedirs(
        "models",
        exist_ok=True
    )


    model_path = "models/model.pkl"



    joblib.dump(
        model,
        model_path
    )


    print("Model Saved")



    # =================================
    # 9. Log Model to MLflow
    # =================================


    mlflow.sklearn.log_model(
        model,
        "teacher_model"
    )


print("MLflow Run Completed")
