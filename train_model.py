import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_text(text):
    # Basic preprocessing: lowercasing (TfidfVectorizer does this by default, but we can add more if needed)
    return text

def train_and_evaluate():
    # Load dataset
    df = load_data("emails.csv")
    
    X = df['text']
    y = df['label']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create a pipeline that first extracts TF-IDF features and then trains a Random Forest Classifier
    # TfidfVectorizer is great for text, it will capture important keywords and URL tokens.
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', preprocessor=preprocess_text),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Phishing', 'Safe'])
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Confusion Matrix:")
    print("True \\ Pred | Phishing | Safe")
    print("-" * 35)
    print(f"Phishing    | {conf_matrix[0][0]:<8} | {conf_matrix[0][1]}")
    print(f"Safe        | {conf_matrix[1][0]:<8} | {conf_matrix[1][1]}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    trained_model = train_and_evaluate()
    print("Model training complete. Ready for predictions!")
    
    # Example prediction
    sample_emails = [
        "Your account needs verification. Click here: http://verify-now.com",
        "Hi, are we still meeting for lunch at 12?"
    ]
    
    print("\n--- Example Predictions ---")
    for email in sample_emails:
        prediction = trained_model.predict([email])[0]
        print(f"Email: '{email}'")
        print(f"Prediction: {prediction}\n")
