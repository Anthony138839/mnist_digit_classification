# MNIST Digit Classification

This project is a simple implementation of a machine learning model to classify handwritten digits (0–9) using the MNIST dataset. It was created as a beginner's project to learn the basics of deep learning and image classification using TensorFlow and Keras.

## 🧠 What This Project Does

- Loads the MNIST dataset (70,000 grayscale images of handwritten digits).
- Preprocesses the data by normalizing pixel values and reshaping.
- Builds a Convolutional Neural Network (CNN) model using TensorFlow/Keras.
- Trains the model on the training data.
- Evaluates the model on the test set.
- Makes predictions and visualizes the results.

## 📁 Project Structure

- `mnist_digit_classification.ipynb`: The main Jupyter notebook containing the code for data loading, preprocessing, model building, training, evaluation, and prediction.

## 🛠️ Requirements

Install the following Python libraries:

```bash
pip install tensorflow matplotlib numpy
```

You also need Jupyter Notebook or JupyterLab to run the `.ipynb` file.

## 🚀 How to Run the Project

1. Clone the repository or download the notebook file.
2. Open the notebook using Jupyter:
   ```bash
   jupyter notebook mnist_digit_classification.ipynb
   ```
3. Run the cells step by step to:
   - Load and visualize the data.
   - Build and train the model.
   - Evaluate its performance.
   - Test it on new images from the dataset.

## 📊 Model Overview

- **Input shape**: 28x28 grayscale images.
- **Model type**: CNN (Convolutional Neural Network).
- **Optimizer**: Adam
- **Loss function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## 📈 Results

- Achieved over **98% accuracy** on the test dataset.
- Demonstrated correct predictions on example images.

## 📚 What I Learned

This project helped me understand:
- How to work with image data in Python.
- The architecture of CNNs.
- Training and evaluating deep learning models using Keras.
- How to visualize model predictions.

## ✅ To-Do / Improvements

- Add more layers to experiment with accuracy.
- Try dropout for regularization.
- Convert the notebook to a script or app.
- Test on custom handwritten digit images.

---

Feel free to fork this project or use it as a template for your own digit recognition experiments!
