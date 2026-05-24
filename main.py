from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("student_model.pkl")

# Input data structure
class StudentData(BaseModel):
    hours: float

# Home route
@app.get("/")
def home():
    return {
        "message": "Student Marks Prediction API"
    }

# Prediction route
@app.post("/predict")
def predict(data: StudentData):

    # Create dataframe for prediction
    input_data = pd.DataFrame({
        "hours": [data.hours]
    })

    # Predict marks
    prediction = model.predict(input_data)

    # Return response
    return {
        "hours_studied": data.hours,
        "predicted_marks": prediction[0]
    }