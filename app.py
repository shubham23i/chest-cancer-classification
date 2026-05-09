from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decode_image
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables to reduce TensorFlow memory footprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        # LOAD MODEL ONCE AT STARTUP
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decode_image(image, clApp.filename)
    
    # Use the pre-initialized classifier
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    # Render uses environment variables for Port
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)