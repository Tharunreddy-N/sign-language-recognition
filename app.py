import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the model from the .h5 file
model = load_model("my_model.h5")

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 128, 128, 1)
    return feature / 255.0

# Define labels
label = {0: 'lung_aca', 1: 'lung_n', 2: 'lung_scc'}


# Function for image classification
def predict_image(image):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the image to grayscale if it's in color
    image_gray = image_array

    # Resize and normalize the image
    image_resized = cv2.resize(image_gray, (128, 128))
    image_normalized = image_resized / 255.0

    # Add channel dimension
    image_final = np.expand_dims(image_normalized, axis=0)
    image_final = np.expand_dims(image_final, axis=3)

    # Make prediction
    pred = model.predict(image_final)
    prediction_label = label[np.argmax(pred)]
    return prediction_label

# Streamlit app
def main():
    st.title("Sign Language Detection")
    st.write("Upload an image for classification.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify Image"):
            predicted_class = predict_image(image)
            st.success(f"Predicted Sign: {predicted_class}")

if __name__ == "__main__":
    main()
