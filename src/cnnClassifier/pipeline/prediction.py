import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

        # Absolute model path
        self.model_path = os.path.abspath(
            os.path.join("artifacts", "training", "model.keras")
        )

        print("Loading model from:", self.model_path)

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("Model loaded successfully")
        else:
            self.model = None
            print("Model file not found")

    def predict(self):

        if self.model is None:
            return [{"image": "Model file not found"}]

        try:
            # Load and preprocess image
            img = load_img(self.filename, target_size=(224, 224))

            img_array = img_to_array(img)

            # Normalize
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = self.model(img_array, training=False)

            # Convert tensor to numpy
            score = predictions.numpy()[0][0]

            print("Prediction score:", score)

            # Binary classification
            if score > 0.5:
                prediction = "Normal"
            else:
                prediction = "Adenocarcinoma Cancer"

            return [{"image": prediction}]

        except Exception as e:
            return [{"image": str(e)}]