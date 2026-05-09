import os
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
        self.model_path = os.path.join("artifacts", "training", "model.keras")

    def predict(self):
        
        if not os.path.exists(self.model_path):
            return f"Error: Model file not found at {self.model_path}"

        
        model = tf.keras.models.load_model(self.model_path)

        test_image = load_img(self.filename, target_size=(224, 224))
        test_image = img_to_array(test_image)
        
        
        test_image = test_image.astype('float32') / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)

        
        tf.keras.backend.clear_session()

        if result[0][0] > 0.5:
            prediction = "normal"
        else:
            prediction = "carcinoma"

        return prediction