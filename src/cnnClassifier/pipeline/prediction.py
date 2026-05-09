import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array

# LOAD MODEL ONLY ONCE
MODEL_PATH = os.path.join(
    os.getcwd(),
    "artifacts",
    "training",
    "model.keras"
)

print("Loading model from:", MODEL_PATH)
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

model = load_model(MODEL_PATH)

print("✅ Model loaded successfully")


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        try:

            img = load_img(self.filename, target_size=(224, 224))

            img_array = img_to_array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)

            print("Predictions:", predictions)

            class_names = [
                "Adenocarcinoma Cancer",
                "Large Cell Carcinoma Cancer",
                "Normal",
                "Squamous Cell Carcinoma Cancer"
            ]

            predicted_class = np.argmax(predictions, axis=1)[0]

            result = class_names[predicted_class]

            return [{"image": result}]

        except Exception as e:

            print("🔥 Prediction Error:", str(e))

            return [{"image": str(e)}]