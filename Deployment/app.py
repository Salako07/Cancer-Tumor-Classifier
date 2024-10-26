from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from urllib.parse import quote as url_quote  # Use this as a replacement in your code

# Load your model
model = tf.keras.models.load_model("model.h5")

# Define your image preprocessing function
def preprocess_image(image):
    # Assuming image input as a PIL image and resizing/normalizing as needed
    image = image.resize((224, 224))  # Resize to model's expected input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    try:
        # Read the image file and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
