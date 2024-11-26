import argparse
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Adding arguments to let the script know what we want it to do
parser = argparse.ArgumentParser(description="Unredactor: Train, evaluate, and predict unredacted names.")
parser.add_argument("--mode", required=True, choices=["preprocess", "train", "evaluate", "predict"], 
                    help="Choose what part of the program to run.")
parser.add_argument("--input", help="Input file path for 'predict' mode (e.g., test.tsv).")
parser.add_argument("--output", help="Output file path for 'predict' mode (e.g., submission.tsv).")
args = parser.parse_args()

# File locations
DATA_FILE = "data/unredactor.tsv"  # This is where the training data lives
VECTORIZER_FILE = "vectorizer.pkl"  # Saves the text vectorizer
MODEL_FILE = "unredactor_model.pkl"  # Saves the trained ML model

def preprocess_data():
    """Reads the data file and prepares it for training and validation."""
    print("Preprocessing data...")
    try:
        # Load the dataset (tab-separated)
        data = pd.read_csv(DATA_FILE, sep="\t", on_bad_lines="skip", engine="python")
        # Make sure the columns are named correctly
        data.columns = ["split", "name", "context"]
        # Drop rows that are incomplete
        data = data.dropna(subset=["split", "name", "context"])
        print("Preprocessing completed!")
        return data
    except FileNotFoundError:
        print(f"Error: The data file '{DATA_FILE}' was not found.")
        exit(1)

def train_model(data):
    """Trains a logistic regression model using the training dataset."""
    print("Training model...")
    # Filter only the training data
    train_data = data[data["split"] == "training"]
    X_train, y_train = train_data["context"], train_data["name"]
    
    # Convert the text data into numbers using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train a simple logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Save the vectorizer and model so we can use them later
    with open(VECTORIZER_FILE, "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model, model_file)
    print("Training completed!")

def evaluate_model(data):
    """Tests the model using the validation dataset and shows performance."""
    print("Evaluating model...")
    
    # Make sure the results folder exists to save the metrics
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Load the saved vectorizer and model
    with open(VECTORIZER_FILE, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open(MODEL_FILE, "rb") as model_file:
        model = pickle.load(model_file)
    
    # Filter only the validation data
    validation_data = data[data["split"] == "validation"]
    X_validation, y_validation = validation_data["context"], validation_data["name"]
    X_validation_tfidf = vectorizer.transform(X_validation)
    
    # Use the model to predict names and check how well it does
    y_pred = model.predict(X_validation_tfidf)
    report = classification_report(y_validation, y_pred, zero_division=0)
    print("Evaluation Metrics:\n", report)
    
    # Save the evaluation results to a file
    with open("results/evaluation_metrics.txt", "w") as metrics_file:
        metrics_file.write(report)

def predict_unredaction(input_file, output_file):
    """Uses the trained model to predict names in a new test file."""
    print("Predicting unredacted names...")
    try:
        # Read the test file
        test_data = pd.read_csv(input_file, sep="\t", names=["id", "context"])
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        exit(1)
    
    # Load the saved vectorizer and model
    with open(VECTORIZER_FILE, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open(MODEL_FILE, "rb") as model_file:
        model = pickle.load(model_file)
    
    # Convert the test data's context column into TF-IDF format
    X_test_tfidf = vectorizer.transform(test_data["context"])
    # Use the model to predict names
    predicted_names = model.predict(X_test_tfidf)
    
    # Save the predictions in the required format
    submission_data = pd.DataFrame({"id": test_data["id"], "name": predicted_names})
    submission_data.to_csv(output_file, sep="\t", index=False)
    print(f"Predictions saved to {output_file}.")

if __name__ == "__main__":
    # Figure out what mode the script should run in based on the input
    if args.mode == "preprocess":
        # Run the preprocessing step
        data = preprocess_data()
        print(data.head())  # Show the first few rows for debugging
    elif args.mode == "train":
        # Train the model using the processed data
        data = preprocess_data()
        train_model(data)
    elif args.mode == "evaluate":
        # Evaluate the model using the validation data
        data = preprocess_data()
        evaluate_model(data)
    elif args.mode == "predict":
        # Predict names using a test file
        if not args.input or not args.output:
            print("Error: '--input' and '--output' are required for 'predict' mode.")
        else:
            predict_unredaction(args.input, args.output)
    else:
        print("Invalid mode. Use --help for guidance.")
