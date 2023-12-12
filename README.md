# Convolutional Neural Networks (CNNs) with Iris Dataset

This repository contains an example of a Convolutional Neural Network (CNN) implementation using the widely known Iris dataset. CNNs are powerful tools in the field of computer vision, capable of learning hierarchical representations from structured data.

## Overview

Convolutional Neural Networks leverage convolutional layers, which are essential for filtering and feature extraction. These layers apply learnable filters to input data, capturing local patterns and spatial relationships. In this example, we demonstrate the application of CNNs to the Iris dataset, a classic dataset in machine learning.

## Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

You can install them using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

## Getting Started

1. Clone this repository:

```bash
git clone [https://github.com/Bhargav6031/DL-Assignment-7thsem]
cd iris-cnn
```

2. Run the Jupyter notebook:

```bash
jupyter notebook
```

Open the "cnns-using-iris-dataset.ipynb" notebook and execute each cell to see the step-by-step process of implementing the CNN on the Iris dataset.

## Dataset

The Iris dataset is included in the repository as "IRIS.csv," sourced from the [Iris Flower Dataset](https://www.kaggle.com/arshid/iris-flower-dataset) on Kaggle.

## Implementation Steps

1. Import the necessary libraries.
2. Read the Iris dataset.
3. Explore and analyze the dataset.
4. Separate features and target variables.
5. Encode categorical data.
6. Split the data into training and testing sets.
7. Define the CNN model architecture.
8. Compile the model.
9. Train the model.
10. Evaluate the model on the test data.
11. Plot the training and validation accuracy over epochs.

Feel free to modify the notebook to experiment with different architectures, hyperparameters, or datasets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Iris Flower Dataset](https://www.kaggle.com/arshid/iris-flower-dataset) on Kaggle for providing the dataset.
- TensorFlow and scikit-learn for their powerful tools and libraries in machine learning.

