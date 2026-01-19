from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# ---------------- DATA ----------------
def prepare_data():
    if not os.path.exists("dataset.csv"):
        data = {
            "Study_Hours": [5,4,2,9,12,15,8,20,15,22,18],
            "Attendance": [90,46,75,86,100,95,98,95,98,88,99],
            "Previous_Score": [85,76,65,88,95,85,90,92,98,99,89],
            "Assignment_Score": [14,19,18,15,19,18,19,20,18,16,20],
            "Internal_Marks": [35,32,15,37,38,39,40,40,39,19,40],
            "Final_Performance": [61.5,54.5,42.3,63.3,67.4,62.5,66.4,65.7,68.8,58.4,64.7]
        }
        pd.DataFrame(data).to_csv("dataset.csv", index=False)

    return pd.read_csv("dataset.csv")


# ---------------- MODEL ----------------
def train_model():
    df = prepare_data()

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
    acc = r2_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {acc:.2%}")

    return model


if os.path.exists("model.joblib"):
    model = joblib.load("model.joblib")
else:
    model = train_model()


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        float(data["studyHours"]),
        float(data["attendance"]),
        float(data["prevScore"]),
        float(data["assignmentScore"]),
        float(data["internalMarks"])
    ]]

    prediction = model.predict(features)[0]
    prediction = max(0, min(100, prediction))

    if prediction >= 90:
        grade = "Excellent"
    elif prediction >= 80:
        grade = "Good"
    elif prediction >= 70:
        grade = "Average"
    else:
        grade = "Poor"

    return jsonify({
        "prediction": round(prediction, 1),
        "grade": grade
    })


@app.route("/get_data")
def get_data():
    df = prepare_data()
    return df.to_dict("records")


if __name__ == "__main__":
    app.run()

