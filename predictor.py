import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load pre-trained model
model = load_model('model/crop_disease_model.h5')

# Example disease classes (adjust based on your model)
classes = ['Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Healthy']

def predict_disease(img_path):
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))           # Model input size
    img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return classes[class_idx], float(np.max(prediction))

if __name__ == "__main__":
    img_path = input("Enter path to leaf image: ")
    disease, confidence = predict_disease(img_path)
    print(f"Disease: {disease}")
    print(f"Confidence: {confidence*100:.2f}%")
