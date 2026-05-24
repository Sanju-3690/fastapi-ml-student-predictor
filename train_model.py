import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Create dataset
df = pd.DataFrame({
    "hours":[1,2,3,4,5],
    "marks":[20,40,60,80,100]
})

# Input feature
X = df[["hours"]]

# Output target
y = df["marks"]

# Create model
model = LinearRegression()

# Train model
model.fit(X,y)

# Save model
joblib.dump(model,"student_model.pkl")

print("Model trained successfully")