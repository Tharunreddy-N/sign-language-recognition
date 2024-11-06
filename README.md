# Sign Language Recognition (Indian)

This repository contains code and models for recognizing Indian Sign Language (ISL) gestures using deep learning. The project includes various components, such as data collection, model training, real-time detection, and a Streamlit-based user interface.

## Table of Contents
- Introduction
- Project Structure
- Getting Started
- Data Collection
- Model Training
- Real-Time Detection
- User Interface
- Contributing
- License

## Introduction
Sign language recognition is an important application that enables communication between hearing-impaired individuals and others. This project focuses on recognizing ISL gestures using deep learning techniques.

## Project Structure
The repository is organized as follows:

- `models/`: Contains pre-trained model files (HDF5 format) for gesture recognition.
- `app.py`: A Streamlit-based web application for visualizing gesture recognition.
- `collectadata.py`: Collects data by capturing images of signs from the camera and storing them in alphabet-specific folders (48x48 grayscale images).
- `detect.py`: Performs real-time gesture detection using the trained models.
- `model.ipynb`: Jupyter Notebook containing the deep learning model architecture and training process.
- `split.py`: Splits the dataset into training and testing subsets.

## Getting Started
1. Clone this repository: `git clone https://github.com/Shivasai2004/sign-language-recognition.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Follow the instructions in each script to use the corresponding functionality.

## Data Collection
To collect data for training, run `collectadata.py`. Show the sign in front of the camera and press the corresponding alphabet key. The images will be saved in the appropriate alphabet folder.

## Model Training
Refer to `model.ipynb` for details on the deep learning model architecture and training process.

## Real-Time Detection
Execute `detect.py` to perform real-time gesture detection using the trained models.

## User Interface
Launch the Streamlit app using `streamlit run app.py`. The web interface allows users to interact with the gesture recognition system.

## Contributing
Contributions are welcome! Please follow the contribution guidelines.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
GitHub Repository
