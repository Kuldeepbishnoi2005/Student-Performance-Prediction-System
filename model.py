import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("dataset.csv")

X = df[[
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Internal_Marks"
]]
y = df["Final_Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")

print("Accuracy:", r2_score(y_test, model.predict(X_test)))
