import pandas as pd
import numpy as np
import os

# Ensure datasets directory exists
if not os.path.exists("datasets"):
    os.makedirs("datasets")

# ✅ Preprocess data for Polynomial Regression (House Price Prediction)
def preprocess_polynomial_regression():
    np.random.seed(42)
    square_footage = np.random.randint(500, 5000, 100)
    price = square_footage * 150 + np.random.randint(10000, 50000, 100)

    df = pd.DataFrame({
        "SquareFootage": square_footage,
        "SquareFootage^2": square_footage ** 2,  # Polynomial feature
        "Price": price
    })

    # Save preprocessed dataset
    df.to_csv("datasets/polynomial_regression.csv", index=False)
    return df

# ✅ Preprocess data for KNN Classification (Expensive or Affordable)
def preprocess_knn_classification():
    np.random.seed(42)
    square_footage = np.random.randint(500, 5000, 100)
    bedrooms = np.random.randint(1, 5, 100)
    location_score = np.random.randint(1, 10, 100)
    expensive = (square_footage > 3000) & (location_score > 5)

    df = pd.DataFrame({
        "SquareFootage": square_footage,
        "Bedrooms": bedrooms,
        "LocationScore": location_score,
        "Expensive": expensive.astype(int)  # 1 = Expensive, 0 = Affordable
    })

    # Save preprocessed dataset
    df.to_csv("datasets/knn_classification.csv", index=False)
    return df

# ✅ Preprocess data for Logistic Regression (Sold within 30 Days)
def preprocess_logistic_regression():
    np.random.seed(42)
    square_footage = np.random.randint(500, 5000, 100)
    price = square_footage * 150 + np.random.randint(10000, 50000, 100)
    market_demand_score = np.random.randint(1, 10, 100)
    sold_within_30_days = (market_demand_score > 5) & (price < 600000)

    df = pd.DataFrame({
        "SquareFootage": square_footage,
        "Price": price,
        "MarketDemandScore": market_demand_score,
        "SoldWithin30Days": sold_within_30_days.astype(int)  # 1 = Sold, 0 = Not Sold
    })

    # Save preprocessed dataset
    df.to_csv("datasets/logistic_regression.csv", index=False)
    return df

# ✅ Execute preprocessing and save all datasets
if __name__ == "__main__":
    preprocess_polynomial_regression()
    preprocess_knn_classification()
    preprocess_logistic_regression()
    print("Datasets preprocessed and saved successfully!")
