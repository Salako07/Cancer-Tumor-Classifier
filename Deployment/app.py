from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("Best_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Preprocess and make prediction
    prediction = model.predict(data["input_data"])
    return jsonify({"prediction": prediction.tolist()})
