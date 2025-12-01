# 🔢 MNIST Digit Classifier

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

A state-of-the-art **Convolutional Neural Network (CNN)** implementation for handwritten digit recognition using the MNIST dataset. This project demonstrates proficiency in deep learning, computer vision, and production-ready machine learning systems.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Technical Details](#-technical-details)
- [Examples](#-examples)
- [Future Goals](#-future-goals)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Overview

The MNIST Digit Classifier is a deep learning project that classifies handwritten digits (0-9) with high accuracy. Built using **TensorFlow** and **Keras**, this project showcases:

- **Convolutional Neural Network (CNN)** architecture for optimal image recognition
- **Production-ready** inference pipeline for real-world digit classification
- **Flexible preprocessing** that handles both light-on-dark and dark-on-light digit images
- **Jupyter Notebook** environment for model training and experimentation
- **Modular design** for easy integration into larger applications

### Why This Project?

Handwritten digit recognition is a fundamental problem in computer vision and serves as a gateway to understanding:
- Deep learning fundamentals
- Image preprocessing techniques
- CNN architectures
- Model deployment strategies
- Real-world ML application development

## ✨ Key Features

- 🧠 **Advanced CNN Architecture**: Multi-layer convolutional neural network with max pooling and dropout for optimal performance
- 🎨 **Smart Image Preprocessing**: Automatic resizing, normalization, and grayscale conversion
- 🔄 **Flexible Input Handling**: Works with various image formats and color schemes
- 📊 **Confidence Scoring**: Returns probability distributions across all digit classes
- 🎓 **Educational Value**: Clean, well-documented code perfect for learning
- 🚀 **Easy Deployment**: Pre-trained model ready for immediate use
- 📈 **Visualization Tools**: Built-in functions to display preprocessed images

## 🏗️ Architecture

The project utilizes a **Convolutional Neural Network (CNN)** architecture, which is specifically designed for image classification tasks:

### Model Architecture

```
Input Layer (28x28x1)
    ↓
Convolutional Layer 1 (32 filters, 3x3 kernel, ReLU activation)
    ↓
Max Pooling Layer 1 (2x2 pool size)
    ↓
Convolutional Layer 2 (64 filters, 3x3 kernel, ReLU activation)
    ↓
Max Pooling Layer 2 (2x2 pool size)
    ↓
Flatten Layer
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Dropout Layer (0.5 rate)
    ↓
Output Layer (10 neurons, Softmax activation)
```

### Why CNN?

Convolutional Neural Networks are ideal for this task because they:
- **Extract spatial features** through convolutional layers
- **Reduce dimensionality** while preserving important information
- **Handle translation invariance** - recognizing digits regardless of position
- **Require fewer parameters** compared to fully connected networks
- **Achieve superior accuracy** on image classification tasks

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/sbekkam07/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Quick Start - Predict a Digit

```python
from src.predict_digit import predict_digit, show_preprocessed

# Path to your digit image
image_path = "examples/example_digit_black_3.png"

# Visualize preprocessed image
show_preprocessed(image_path)

# Make prediction
digit, probabilities = predict_digit(image_path)

print(f"Predicted digit: {digit}")
print(f"Confidence: {probabilities[digit]*100:.2f}%")
```

### Running the Example Script

```bash
python src/predict_digit.py
```

### Training Your Own Model

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/MNIST_Digits_Clasification-2.ipynb
   ```

2. Run all cells to:
   - Load the MNIST dataset
   - Preprocess the data
   - Build the CNN architecture
   - Train the model
   - Evaluate performance
   - Save the trained model

### Using Custom Images

The classifier can handle custom digit images:

```python
from src.predict_digit import predict_digit

# Your custom image
my_image = "path/to/your/digit.png"
digit, probs = predict_digit(my_image)

print(f"Predicted: {digit}")
```

**Image Requirements:**
- Any size (will be resized to 28x28)
- Grayscale or color (will be converted to grayscale)
- For dark digits on white background, uncomment line 21 in `predict_digit.py`

## 📁 Project Structure

```
mnist-digit-classifier/
│
├── src/
│   └── predict_digit.py      # Main inference script
│
├── models/
│   ├── mnist_cnn_model.h5    # Trained CNN model (primary)
│   └── mnist_model.h5         # Alternative model
│
├── notebooks/
│   └── MNIST_Digits_Clasification-2.ipynb  # Training notebook
│
├── examples/
│   ├── example_digit.png              # Sample digit images
│   ├── example_digit_black_3.png      # for testing
│   ├── example_digit_black_4.png
│   ├── example_digit_black_6.png
│   ├── example_digit_black_6_2.png
│   └── example_digit_white_6.png
│
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## 📊 Model Performance

The CNN model achieves impressive results on the MNIST dataset:

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98-99%
- **Inference Time**: <50ms per image
- **Model Size**: 2.7 MB (highly portable)

### Dataset Information

- **Training Samples**: 60,000 images
- **Test Samples**: 10,000 images
- **Image Size**: 28x28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

## 🔬 Technical Details

### Preprocessing Pipeline

1. **Load Image**: Read image file using PIL
2. **Grayscale Conversion**: Convert to single-channel grayscale
3. **Resize**: Standardize to 28x28 pixels
4. **Normalization**: Scale pixel values to [0, 1] range
5. **Reshape**: Format to (1, 28, 28, 1) for CNN input
6. **Optional Inversion**: Handle different background colors

### Training Process

1. **Data Loading**: Load MNIST dataset from Keras
2. **Data Preprocessing**: Normalize and reshape images
3. **Model Compilation**: 
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy
4. **Training**: Multiple epochs with validation split
5. **Model Saving**: Export trained model in HDF5 format

### Technology Stack

- **Deep Learning Framework**: TensorFlow 2.x / Keras
- **Numerical Computing**: NumPy
- **Image Processing**: PIL (Pillow)
- **Visualization**: Matplotlib
- **Development Environment**: Jupyter Notebook
- **Version Control**: Git

## 🎨 Examples

The `examples/` directory contains various test images:

- **White background digits**: `example_digit_white_6.png`
- **Black background digits**: `example_digit_black_*.png`
- Various handwriting styles and thicknesses

Try them all to see the model's robustness!

## 🚀 Future Goals

This project has exciting potential for expansion. Here are planned enhancements:

### 🌐 Full-Stack Web Application
- **Frontend**: React or Vue.js interface with canvas drawing functionality
- **Backend**: Flask/FastAPI REST API for model serving
- **Features**:
  - Real-time digit drawing and recognition
  - Confidence visualization with bar charts
  - Image upload functionality
  - Historical predictions tracking
  - Responsive mobile-friendly design

### 📱 Mobile Application
- Develop iOS/Android app using TensorFlow Lite
- Camera integration for capturing handwritten digits
- Offline prediction capability

### 🔧 Model Enhancements
- Implement data augmentation for improved robustness
- Experiment with deeper architectures (ResNet, DenseNet)
- Add support for multiple digit recognition
- Fine-tune hyperparameters for optimal performance

### 🐳 Deployment & DevOps
- Dockerize the application for easy deployment
- Set up CI/CD pipeline with GitHub Actions
- Deploy to cloud platforms (AWS, GCP, or Azure)
- Implement model versioning and A/B testing

### 📊 Extended Features
- Add explainability using Grad-CAM visualizations
- Support for other handwriting datasets (EMNIST, Kuzushiji)
- Batch prediction API
- Performance benchmarking suite
- Interactive model training dashboard

### 🤝 Community Features
- Create web interface for community contributions
- Build a dataset of user-submitted digits
- Leaderboard for model accuracy
- Open API for third-party integrations

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Update documentation for new features
- Test your changes thoroughly

## 📄 License

This project is open source and available for educational and personal use. Feel free to fork, modify, and use this project for your own learning and development.

## 📧 Contact

**Your Name** - sbekkam07

- GitHub: [@sbekkam07](https://github.com/sbekkam07)
- Project Link: [https://github.com/sbekkam07/mnist-digit-classifier](https://github.com/sbekkam07/mnist-digit-classifier)

---

### 🌟 Acknowledgments

- **MNIST Database**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow/Keras**: Google Brain team and contributors
- **Python Community**: For excellent libraries and tools

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐!**

Made with ❤️ and lots of ☕

</div>