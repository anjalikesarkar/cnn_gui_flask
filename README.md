# CNN + PyQt + Flask Trial Project (CIFAR-10)

A demo project combining a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset, a PyQt5 GUI for local desktop interaction, and a Flask web app for browser-based access.

# Integration
- The PyQt GUI calls the CNN model for inference on images loaded via the interface
- The Flask web app exposes APIs and frontend pages to upload images and show CNN predictions

# Setup: 
* main.py
  - This is the core training script for the CNN model on CIFAR-10
  - Hyperparameters configuration
  - Device detection (CPU or GPU)
  - Data import and loading using CIFAR-10 dataset
  - Training loop for model optimization
  - Save model

* model.py
  - This file defines the CNN architecture used for CIFAR-10 classification
  - Includes: 3 Conv layers, ReLU activation function and maxpooling + flatten and linear layer

* predict.py
  - Load the saved CNN model weights
  - Apply the same transformations to test images or any input image
  - Run inference and output predictions
 
* app.py
  - PyQt5 that sets up a simple GUI with three buttons:
  - Select — to select an image file
  - Predict — to send the image to your Flask backend for prediction
  - Exit — to close the application

* server.py
  - The /predict route accepts POST requests with an image file named 'file'.
  - The image is processed and transformed the same way as training images.
  - The model predicts the class, which is sent back as JSON to your PyQt GUI.



