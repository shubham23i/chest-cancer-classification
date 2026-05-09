from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import os
import base64
import tensorflow as tf
from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

# 1. Load the model globally at startup to save memory on Render
model_path = os.path.join("artifacts", "training", "model.keras")
model = None
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Model file not found at {model_path}")

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

client_app = ClientApp()

@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def PredictRoute():
    try:
        image = request.json['image']
        decodeImage = base64.b64decode(image)

        with open(client_app.filename, "wb") as f:
            f.write(decodeImage)

        # 2. Pass the pre-loaded model to the predict method
        result = client_app.classifier.predict(model)

        return jsonify({"prediction": result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)