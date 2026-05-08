import os
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        model = tf.keras.models.load_model(
            os.path.join("artifacts", "training", "model.keras")
        )

        test_image = load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = img_to_array(test_image)

        test_image = test_image / 255.0

        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)

        if result[0][0] > 0.5:
            prediction = "normal"
        else:
            prediction = "carcinoma"

        return prediction