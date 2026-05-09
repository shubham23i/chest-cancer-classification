def predict(self):
    try:
        if self.model is None:
            return [{"image": "Model file not found"}]

        img = load_img(self.filename, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype("float32") / 255.0

        predictions = self.model(img_array, training=False)

        print("Raw predictions:", predictions)

        score = predictions.numpy()[0][0]

        if score > 0.5:
            prediction = "Normal"
        else:
            prediction = "Adenocarcinoma Cancer"

        return [{"image": prediction}]

    except Exception as e:
        print("🔥 Prediction error:", str(e))
        return [{"image": str(e)}]