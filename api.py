from flask import Blueprint, jsonify, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Creating a Blueprint for the API routes
bp = Blueprint('api', __name__)

@bp.route('/')
def home():
    return jsonify({
        "message": "Welcome to Genepattern API",
        "endpoints": {
            "/api/uploadCsv": "POST - Upload and process CSV data",
            "/api/generateImage": "POST - Generate visualization from CSV data",
            "/api/predict": "POST - Train KNN model and get predictions"
        }
    })


# Route to upload and process CSV data
@bp.route('/api/uploadCsv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Read the CSV file into a DataFrame
    try:
        data = pd.read_csv(file)
        # Get basic summary statistics
        summary = data.describe().to_dict()
        return jsonify({"message": "CSV processed successfully", "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# route to generate an image with a Matplotlib plot from CSV data
@bp.route('/api/generateImage', methods=['POST'])
def generate_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        data = pd.read_csv(file)

        # Create a sample plot using Matplotlib
        plt.figure(figsize=(8, 6))
        if data.shape[1] >= 2:  # Ensure at least two columns exist
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
            plt.xlabel(data.columns[0])
            plt.ylabel(data.columns[1])
            plt.title("Sample Scatter Plot")
        else:
            plt.text(0.5, 0.5, "Insufficient data for plotting", ha='center')

        # Save the plot to a buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()  # Close the plot to free up memory

        return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to train a KNN model and return predictions
@bp.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        data = pd.read_csv(file)
        if data.shape[1] < 2:
            return jsonify({"error": "Data must have at least two columns for prediction"}), 400

        # Extract features and target for KNN model
        X = data.iloc[:, :-1].values  # Features
        y = data.iloc[:, -1].values   # Target

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return jsonify({
            "message": "KNN model trained successfully",
            "accuracy": accuracy,
            "predictions": predictions[:10].tolist()  # Return the first 10 predictions for review
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

        