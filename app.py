from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import os
import base64

from cnnClassifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline(
            filename="inputImage.jpg"
        )

client_app = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def PredictRoute():
    image = request.json['image']

    decodeImage = base64.b64decode(image)

    with open("inputImage.jpg", "wb") as f:
        f.write(decodeImage)

    result = client_app.classifier.predict()

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)