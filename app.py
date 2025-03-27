from flask import Flask, render_template, request
import joblib
import os
from models import train_all_models, predict_polynomial_regression, predict_knn_classifier, predict_logistic_regression

app = Flask(__name__)

# Ensure models are trained if not already done.
if not os.path.exists("models/polynomial_regression.pkl"):
    accuracies = train_all_models()
else:
    # Optionally, retrain and get updated accuracy scores.
    accuracies = train_all_models()

# Home route
@app.route("/")
def home():
    return render_template("index.html", accuracy_scores=accuracies, active_page="home")

# Polynomial Regression route
@app.route("/polynomial", methods=["GET", "POST"])
def polynomial():
    prediction = None
    if request.method == "POST":
        square_footage = float(request.form["square_footage"])
        prediction = predict_polynomial_regression(square_footage)
    return render_template("polynomial.html", prediction=prediction, active_page="polynomial")

# KNN Classifier route
@app.route("/knn", methods=["GET", "POST"])
def knn():
    prediction = None
    if request.method == "POST":
        square_footage = float(request.form["square_footage"])
        bedrooms = int(request.form["bedrooms"])
        location_score = int(request.form["location_score"])
        pred = predict_knn_classifier(square_footage, bedrooms, location_score)
        prediction = "Expensive" if pred == 1 else "Affordable"
    return render_template("knn.html", prediction=prediction, active_page="knn")

# Logistic Regression route
@app.route("/logistic", methods=["GET", "POST"])
def logistic():
    prediction = None
    if request.method == "POST":
        square_footage = float(request.form["square_footage"])
        price = float(request.form["price"])
        market_demand_score = int(request.form["market_demand_score"])
        pred = predict_logistic_regression(square_footage, price, market_demand_score)
        prediction = "Sold within 30 days" if pred == 1 else "Not Sold"
    return render_template("logistic.html", prediction=prediction, active_page="logistic")

if __name__ == "__main__":
    app.run(debug=False)
