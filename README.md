# Phishing Email Detection Model

This project implements a machine learning model using Scikit-Learn to classify emails as either "Phishing" or "Safe". It uses a Random Forest Classifier trained on text features extracted via TF-IDF (Term Frequency-Inverse Document Frequency) which captures important keywords and URL structures in the emails.

## Features

- **Training Pipeline**: Automatically loads data, splits it into training/testing sets, and builds a feature extraction + classification pipeline.
- **Evaluation**: Displays accuracy, confusion matrix, and a detailed classification report.
- **Prediction**: Includes an easy way to predict the nature of new, out-of-sample emails.

## Project Structure

- `emails.csv`: The dataset containing examples of safe and phishing emails (includes deceptive URLs and keywords).
- `train_model.py`: The main Python script that trains and evaluates the model.
- `requirements.txt`: Python package dependencies.

## Installation

Ensure you have Python installed, then run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

To train the model and see the evaluation metrics and predictions, run:

```bash
python train_model.py
```

### Expected Output

The script will output the accuracy score, a confusion matrix showing the true vs. predicted values, and sample predictions for test emails.

## Next Steps

To improve or scale this model, you can replace the contents of `emails.csv` with a much larger dataset of real-world emails. The pipeline is designed to automatically adapt to the new data.
