import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


class PredictionPipeline:

    def __init__(self, filename):

        self.filename = filename

        self.model_path = os.path.join(
            os.getcwd(),
            "artifacts",
            "training",
            "model.keras"
        )

        print("MODEL PATH:", self.model_path)

        if os.path.exists(self.model_path):
            print("✅ Model file exists")
            self.model = load_model(self.model_path)
            print("✅ Model loaded")
        else:
            print("❌ Model file NOT found")
            self.model = None

    def predict(self):

        try:

            print("Image path:", self.filename)

            if not os.path.exists(self.filename):
                return [{"image": "Image file not found"}]

            if self.model is None:
                return [{"image": "Model not loaded"}]

            # Load image
            img = load_img(self.filename, target_size=(224, 224))

            # Convert image to array
            img_array = img_to_array(img)

            print("Original shape:", img_array.shape)

            # Normalize
            img_array = img_array / 255.0

            # Expand dimensions
            img_array = np.expand_dims(img_array, axis=0)

            print("Final input shape:", img_array.shape)

            # Prediction
            predictions = self.model.predict(img_array)

            print("Raw Predictions:", predictions)

            # MULTICLASS SUPPORT
            class_names = [
                "Adenocarcinoma Cancer",
                "Large Cell Carcinoma Cancer",
                "Normal",
                "Squamous Cell Carcinoma Cancer"
            ]

            predicted_class = np.argmax(predictions, axis=1)[0]

            prediction = class_names[predicted_class]

            print("Predicted class:", prediction)

            return [{"image": prediction}]

        except Exception as e:

            print("🔥 PREDICTION ERROR:", str(e))

            return [{"image": str(e)}]