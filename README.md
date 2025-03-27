# ML Flask Real Estate Predictor

A Flask-based web application for real estate predictions. Includes:
- **Polynomial Regression**: Predicts house prices based on square footage.
- **KNN Classifier**: Classifies properties as "Expensive" or "Affordable" based on square footage, bedrooms, and location score.
- **Logistic Regression**: Predicts whether a property will be "Sold within 30 days" based on square footage, price, and market demand score.

## 🚀 Deployment
Deployed on [Render](https://render.com/). Access the live app [here](#) (replace with your Render URL after deployment).

## 🛠️ Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-flask-real-estate-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ml-flask-real-estate-predictor
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app locally:
   ```bash
   python app.py
   ```

## 🌐 Usage
- **Home**: View model accuracy scores.
- **Polynomial Regression**: Predict house prices based on square footage.
- **KNN Classifier**: Classify properties as "Expensive" or "Affordable".
- **Logistic Regression**: Predict whether a property will be "Sold within 30 days".

## 📂 Project Structure
```
ml-flask-real-estate-predictor/
├── app.py                # Main Flask application
├── models.py             # Model definitions and training
├── preprocess.py         # Data preprocessing
├── train_models.py       # Script for training models
├── requirements.txt      # Project dependencies
├── Procfile              # Deployment configuration
├── datasets/             # Preprocessed datasets
├── models/               # Trained models
├── static/               # Static assets
└── templates/            # Flask HTML templates
```

## 📄 License
MIT License. See [LICENSE](#) for details.
