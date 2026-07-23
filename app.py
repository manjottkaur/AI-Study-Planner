from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

df = pd.read_csv("dataset/AI_Study_Planner_Dataset.csv")
df.drop_duplicates(inplace=True)
le_subject = LabelEncoder()
le_academic = LabelEncoder()

df["subject"] = le_subject.fit_transform(df["subject"])
df["academic_level"] = le_academic.fit_transform(df["academic_level"])

X = df[
    [
        "subject",
        "academic_level",
        "exam_score",
        "days_left",
        "available_hours_per_day",
        "difficulty_level",
    ]
]

y = df["recommended_study_hours"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

@app.route("/")
def home():

    return render_template(
        "index.html",
        subjects=list(le_subject.classes_),
        academics=list(le_academic.classes_),
    )

@app.route("/predict", methods=["POST"])
def predict():

    subject = request.form["subject"]
    academic = request.form["academic"]

    exam_score = float(request.form["exam_score"])
    days_left = int(request.form["days_left"])
    available_hours = float(request.form["available_hours"])
    difficulty = int(request.form["difficulty"])

    subject = le_subject.transform([subject])[0]
    academic = le_academic.transform([academic])[0]

    prediction = model.predict(
        [[
            subject,
            academic,
            exam_score,
            days_left,
            available_hours,
            difficulty
        ]]
    )[0]

    return render_template(
        "index.html",
        prediction=round(prediction,2),
        subjects=list(le_subject.classes_),
        academics=list(le_academic.classes_),
    )

if __name__ == "__main__":
    app.run(debug=True)