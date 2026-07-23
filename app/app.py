from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/model.pkl")

# Create FastAPI app
app = FastAPI(
    title="Teacher Performance Prediction API"
)

# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app)


# Input schema
class TeacherData(BaseModel):
    experience: int
    student_feedback: float
    pass_percentage: float
    attendance: float
    research_papers: int
    workshops: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Teacher Performance Prediction API"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: TeacherData):

    input_data = pd.DataFrame(
        [[
            data.experience,
            data.student_feedback,
            data.pass_percentage,
            data.attendance,
            data.research_papers,
            data.workshops
        ]],
        columns=[
            "Experience",
            "Student_Feedback",
            "Pass_Percentage",
            "Attendance",
            "Research_Papers",
            "Workshops"
        ]
    )

    prediction = model.predict(input_data)

    return {
        "Predicted Performance Score": round(float(prediction[0]), 2)
    }