# MNIST Digit Recognition with CNN and Image Augmentation

This repository contains a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to recognize handwritten digits from the MNIST dataset. The model is trained using augmented data to improve its generalization capabilities.

## Requirements

- Python
- TensorFlow
- Keras (included with TensorFlow)
- OpenCV
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib
```
## Overview


### 1. Data Loading:

The MNIST dataset is loaded using TensorFlow's built-in utility. 

To enhance the diversity of the training data, an ImageDataGenerator is used to apply random transformations like rotation, zooming, and shifting.
### 2. Model Architecture:

The CNN model is built with the following layers:
- Two convolutional layers with ReLU activation followed by max-pooling layers.
- A flattening layer to convert the 2D matrix data to a vector.
- A dense layer with 128 units and ReLU activation.
- A dropout layer with a rate of 0.5 to prevent overfitting.
- An output dense layer with 10 units and softmax activation for multi-class classification.
### 3. Model Compilation, Training, and Testing:

The model is compiled using the Adam optimizer and sparse categorical crossentropy as the loss function.
It is trained on the augmented dataset for 10 epochs but can be increased with more computational power.

After training, the model is evaluated on the test dataset to determine its accuracy. On top of the test dataset, you can handwrite numbers in Microsoft Paint and then the script will automatically load these images, preprocess them, and output the predicted digit.
(Ensure your custom images are in grayscale and have a resolution of 28x28 pixels for the best results.)

### 4. Results
Using 10 epochs, the model is accurate 99% of the time.

![image](https://github.com/user-attachments/assets/418f6279-a392-49c9-88a9-a02ddfc381c3)
