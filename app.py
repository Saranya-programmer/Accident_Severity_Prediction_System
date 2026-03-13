from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model and class names
model = tf.keras.models.load_model("accident_severity_model.keras")
class_names = ["Hand", "Head", "Leg"]  # Ensure this matches your training dataset

# Confidence threshold to allow predictions
CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_severity(probability):
    if probability > 0.66:
        return "Severe 🔴"
    elif probability > 0.33:
        return "Moderate 🟡"
    else:
        return "Mild 🟢"

def recommend_hospital(predicted_class):
    hospital_mapping = {
        "Hand": "🏥 Kamineni Hospitals",
        "Head": "🧠 Apollo Hospitals (Brain & Spine)",
        "Leg": "🚑 Manipal Hospital (Critical Injuries)"
    }
    return hospital_mapping.get(predicted_class, "🏨 General Hospital (All Cases)")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    severity = None
    hospital = None
    probabilities = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image = Image.open(file.stream)
            img_array = preprocess_image(image)

            predictions = model.predict(img_array)
            probabilities_raw = tf.nn.softmax(predictions[0]).numpy()

            predicted_index = np.argmax(probabilities_raw)
            confidence_score = probabilities_raw[predicted_index]

            if confidence_score < CONFIDENCE_THRESHOLD:
                error = "❌ Unable to confidently identify injury type. Please upload a clearer injury image."
            else:
                predicted_class = class_names[predicted_index]
                prediction = predicted_class
                confidence = f"{confidence_score * 100:.2f}%"
                severity = get_severity(confidence_score)
                hospital = recommend_hospital(predicted_class)
                probabilities = [(class_names[i], f"{prob * 100:.2f}%") for i, prob in enumerate(probabilities_raw)]


    return render_template("index.html", prediction=prediction, confidence=confidence,
                           severity=severity, hospital=hospital, probabilities=probabilities, error=error)

if __name__ == "__main__":
    app.run(debug=True)
    '''
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model and class names
model = tf.keras.models.load_model("accident_severity_model.keras")
class_names = ["Hand", "Head", "Leg"]  # Ensure this matches your training dataset

# Confidence threshold to allow predictions
CONFIDENCE_THRESHOLD = 0.6

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_severity(probability):
    if probability > 0.66:
        return "Severe 🔴"
    elif probability > 0.33:
        return "Moderate 🟡"
    else:
        return "Mild 🟢"

def recommend_hospital(predicted_class):
    hospital_mapping = {
        "Hand": "🏥 Kamineni Hospitals",
        "Head": "🧠 Apollo Hospitals (Brain & Spine)",
        "Leg": "🚑 Manipal Hospital (Critical Injuries)"
    }
    return hospital_mapping.get(predicted_class, "🏨 General Hospital (All Cases)")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    severity = None
    hospital = None
    probabilities = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image = Image.open(file.stream)
            img_array = preprocess_image(image)

            predictions = model.predict(img_array)
            probabilities_raw = tf.nn.softmax(predictions[0]).numpy()

            predicted_index = np.argmax(probabilities_raw)
            confidence_score = probabilities_raw[predicted_index]

            if confidence_score < CONFIDENCE_THRESHOLD:
                error = "❌ Unable to confidently identify injury type. Please upload a clearer injury image."
            else:
                predicted_class = class_names[predicted_index]
                prediction = predicted_class
                confidence = f"{confidence_score * 100:.2f}%"
                severity = get_severity(confidence_score)
                hospital = recommend_hospital(predicted_class)
                probabilities = [(class_names[i], f"{prob * 100:.2f}%") for i, prob in enumerate(probabilities_raw)]

    return render_template("index.html", prediction=prediction, confidence=confidence,
                           severity=severity, hospital=hospital, probabilities=probabilities, error=error)

if __name__ == "__main__":
    app.run(debug=True)
    '''
