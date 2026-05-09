import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img, img_to_array

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load model once during initialization
        self.model_path = os.path.join("artifacts", "training", "model.keras")
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = None

    def predict(self):
        if self.model is None:
            return [{"image": "Model file not found"}]

        # Preprocessing
        img = load_img(self.filename, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        # Inference - Using __call__ for lower memory overhead
        predictions = self.model(img_array, training=False)
        score = predictions[0][0]

        if score > 0.5:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'

        return [{"image": prediction}]