from keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model("signlanguageb32.h5")

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.reshape(1, 128, 128, 1)
    image = image / 255.0
    return image

# Define the class labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle for the region of interest
    cv2.rectangle(frame, (50, 50), (350, 350), (0, 165, 255), 2)
    
    # Crop the region of interest
    roi = frame[50:350, 50:350]
    
    # Preprocess the region of interest
    preprocessed_roi = preprocess_image(roi)
    
    # Make prediction
    pred = model.predict(preprocessed_roi)
    label_index = np.argmax(pred)
    prediction_label = labels[label_index]
    accuracy = np.max(pred) * 100
    
    # Display the prediction
    cv2.rectangle(frame, (0, 0), (400, 50), (0, 165, 255), -1)
    display_text = f'Prediction: {prediction_label} ({accuracy:.2f}%)'
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow("Sign Language Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
