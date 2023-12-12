**Droupots of DL**

# Iris Flower Classification with Dropout in Neural Network

## Overview

This repository contains a Python script for building a neural network model using TensorFlow and Keras to classify the Iris flower dataset. The model includes dropout layers to enhance generalization and prevent overfitting.

## Project Structure

- **Notebook.ipynb**: Jupyter Notebook containing the code.
- **/data**: Folder to store the dataset.
- **README.md**: Documentation for the project.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Matplotlib
- Pandas
- NumPy
- Scikit-learn

### Installation

1. Clone the repository:

   bash
   git clone https://github.com/your-username/iris-flower-classification.git
   cd iris-flower-classification
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   

3. Open the Jupyter Notebook:

   bash
   jupyter notebook Notebook.ipynb
   

4. Run the notebook cells to execute the code.

## Dataset

The Iris dataset is used for this project. It includes features such as sepal length, sepal width, petal length, and petal width to classify three species of Iris flowers.

## Exploratory Data Analysis

The notebook includes an initial exploration of the dataset, displaying information about the data and the first ten rows.

## Data Preprocessing

- Separation of features and target variable.
- Encoding of the categorical target variable using Label Encoding and one-hot encoding.
- Splitting the data into training and testing sets.

## Model Architecture

The neural network model is defined with the following architecture:

- Input layer
- Dense layer (1000 neurons, ReLU activation)
- Dense layer (500 neurons, ReLU activation)
- Dense layer (300 neurons, ReLU activation)
- Dropout layer (20% dropout rate)
- Output layer (3 neurons, Softmax activation)

## Compilation and Training

The model is compiled using the Adam optimizer and categorical crossentropy loss. It is then trained on the training set with validation on the test set over 30 epochs.

## Evaluation

The model's performance is evaluated on the test set, and the accuracy is printed.

## Dropout

The dropout layer is implemented to prevent overfitting. It randomly drops a specified percentage of neurons during training, forcing the network to learn more robust features.

## Results

The training and validation accuracy over epochs are plotted to visualize the model's performance.

Feel free to modify and experiment with the code to enhance the model or try different datasets.



This README provides an overview of the project, installation instructions, details about the dataset, model architecture, training process, and results. It also explains the concept of dropout and encourages users to experiment with the code.
