# Accident Severity Prediction System

## Problem Statement
Accident injury assessment often requires manual evaluation, which may delay medical response in emergency situations. 
This project aims to assist in identifying injury categories from accident images using a deep learning model, enabling faster preliminary assessment.

---

## Project Overview
The Accident Severity Prediction System is a web-based application that uses a deep learning model to classify accident injury images. 
The system analyzes uploaded images and predicts the type of injury while also providing severity levels and recommended hospitals.
This project demonstrates how machine learning can assist in faster medical assessment during accident situations.

---

## Features
- Upload accident injury images through a web interface
- Classifies injury type (Hand, Head, Leg)
- Displays prediction confidence
- Provides severity level (Mild, Moderate, Severe)
- Recommends hospitals based on predicted injury type
- Displays probability distribution for each injury class

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Flask
- Pillow (PIL)
- HTML (Frontend)

---

## Dataset Classes
The model is trained to classify the following injury categories:

- Hand Injury
- Head Injury
- Leg Injury

---

## Project Structure
Accident_Severity_Prediction_System
│
├── app.py
├── accident_severity_model.keras
├── Train.ipynb
├── requirements.txt
├── templates/
│ └── index.html
├── sample_dataset/
├── screenshots/
└── README.md

---

## How the System Works
1. The user uploads an accident injury image through the web interface.
2. The uploaded image is preprocessed and resized to 128×128 pixels.
3. The trained deep learning model analyzes the image.
4. The model predicts the injury category (Hand, Head, Leg).
5. The system calculates prediction confidence.
6. A severity level is determined based on the prediction probability.
7. The system recommends a hospital based on the predicted injury type.

---

