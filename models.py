import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from preprocess import preprocess_polynomial_regression, preprocess_knn_classification, preprocess_logistic_regression

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# ----- Training Functions -----
def train_polynomial_regression():
    df = pd.read_csv("datasets/polynomial_regression.csv")  # Load preprocessed dataset
    X = df[["SquareFootage", "SquareFootage^2"]]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "models/polynomial_regression.pkl")

    # Calculate accuracy (R² Score)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def train_knn_classifier():
    df = pd.read_csv("datasets/knn_classification.csv")  # Load preprocessed dataset
    X = df[["SquareFootage", "Bedrooms", "LocationScore"]]
    y = df["Expensive"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "models/knn_classifier.pkl")

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_logistic_regression():
    df = pd.read_csv("datasets/logistic_regression.csv")  # Load preprocessed dataset
    X = df[["SquareFootage", "Price", "MarketDemandScore"]]
    y = df["SoldWithin30Days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "models/logistic_regression.pkl")

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train_all_models():
    pr_acc = train_polynomial_regression()
    knn_acc = train_knn_classifier()
    log_acc = train_logistic_regression()
    return {
        "Polynomial Regression R2 Score": pr_acc,
        "KNN Classifier Accuracy": knn_acc,
        "Logistic Regression Accuracy": log_acc
    }

# ----- Prediction Functions -----
def predict_polynomial_regression(square_footage):
    model = joblib.load("models/polynomial_regression.pkl")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    transformed_input = poly.fit_transform([[square_footage]])
    return model.predict(transformed_input)[0]

def predict_knn_classifier(square_footage, bedrooms, location_score):
    model = joblib.load("models/knn_classifier.pkl")
    return model.predict([[square_footage, bedrooms, location_score]])[0]

def predict_logistic_regression(square_footage, price, market_demand_score):
    model = joblib.load("models/logistic_regression.pkl")
    return model.predict([[square_footage, price, market_demand_score]])[0]
