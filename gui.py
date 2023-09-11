"""
Traffic Sign Detection Web Application

This Streamlit web application allows users to upload an image of a traffic sign,
and it uses a pre-trained deep learning model to classify the traffic sign and
display the predicted class along with confidence.

Author: Aditya Chaturvedi
Date: September 11, 2023
"""
import os
import PIL
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Replace with the actual directory where your trained model is located
MODEL_DIR = "/Users/adityachaturvedi/Desktop/7th Sem/Minor Project/CleanedData/"
os.chdir(MODEL_DIR)
model = load_model("traffic_classifier.keras", compile=False)

classes = {
    1: "Speed limit (20km/h)",
    2: "Speed limit (30km/h)",
    3: "Speed limit (50km/h)",
    4: "Speed limit (60km/h)",
    5: "Speed limit (70km/h)",
    6: "Speed limit (80km/h)",
    7: "End of speed limit (80km/h)",
    8: "Speed limit (100km/h)",
    9: "Speed limit (120km/h)",
    10: "No passing",
    11: "No passing veh over 3.5 tons",
    12: "Right-of-way at intersection",
    13: "Priority road",
    14: "Yield",
    15: "Stop",
    16: "No vehicles",
    17: "Veh > 3.5 tons prohibited",
    18: "No entry",
    19: "General caution",
    20: "Dangerous curve left",
    21: "Dangerous curve right",
    22: "Double curve",
    23: "Bumpy road",
    24: "Slippery road",
    25: "Road narrows on the right",
    26: "Road work",
    27: "Traffic signals",
    28: "Pedestrians",
    29: "Children crossing",
    30: "Bicycles crossing",
    31: "Beware of ice/snow",
    32: "Wild animals crossing",
    33: "End speed + passing limits",
    34: "Turn right ahead",
    35: "Turn left ahead",
    36: "Ahead only",
    37: "Go straight or right",
    38: "Go straight or left",
    39: "Keep right",
    40: "Keep left",
    41: "Roundabout mandatory",
    42: "End of no passing",
    43: "End no passing veh > 3.5 tons",
}

# Set up the Streamlit app
st.title("Traffic Sign Detection")
st.sidebar.title("Upload Image")

uploaded_image = st.sidebar.file_uploader(
    "Upload a traffic sign image", type=["jpg", "png", "jpeg"]
)

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform classification when a button is clicked
    if st.sidebar.button("Classify Image"):
        try:
            image = Image.open(uploaded_image)
            image = image.resize((30, 30))
            image = np.expand_dims(image, axis=0)
            image = np.array(image)
            pred_probs = model.predict(image)
            pred_class = np.argmax(pred_probs)
            sign = classes[pred_class + 1]
            confidence = pred_probs[0][pred_class]

            st.write(f"Predicted Traffic Sign: {sign}")
            st.write(f"Confidence: {confidence:.2%}")
        except PIL.UnidentifiedImageError as e:
            st.error(f"Error: {str(e)}")
