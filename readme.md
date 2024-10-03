# Cure My Leaf

**For view the jupyter notebook select the file CureMyLeaf_EN (English)**

**Para ver el archivo jupyter notebook seleciona el archivo CureMyLeaf_ES (ESPAÑOL)**

Cure My Leaf is a convolutional neural network model designed to predict the health status of crop leaves. It can identify if a leaf is affected by Bean bacteria, Angular bacteria, or if it's healthy.

## Table of Contents

- [Cure My Leaf](#cure-my-leaf)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
    - [Activation Functions](#activation-functions)
  - [Deployment](#deployment)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Deploy web](#deploy-web)

## Introduction

Plant diseases can significantly impact crop yields and quality. Cure My Leaf aims to assist farmers and agricultural professionals in early detection of leaf diseases, specifically focusing on bean crops.

## Dataset

The original dataset is sourced from [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) on Hugging Face. You can use this dataset to create your own model or experiment with different approaches.

## Data Preprocessing

We cleaned and preprocessed the original dataset to make it suitable for model training:

    1. Converted image characteristics from base64 format to RGB arrays
    2. Normalized pixel values to the range [0, 1]
    3. Labeled the data into three categories:

0: Angular leaf spot

1: Bean rust

2: Healthy leaf

## Model Training

We utilized several libraries for data handling and model creation:

- Sklearn: For splitting the data into training and testing sets
- Pandas and Numpy: For data manipulation and preprocessing
- TensorFlow: For creating, compiling, and training the neural network

Our model architecture is based on VGG16, which achieved the best performance with an accuracy of `86.82 %%` after approximately `35` training epochs.

### Activation Functions

We used two main activation functions in our model:

1. ReLU (Rectified Linear Unit):

   ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cmax%280%2C%20x%29)

2. Softmax (for the output layer):

   ![equation](https://latex.codecogs.com/gif.latex?%5Csigma%28z%29_i%20%3D%20%5Cfrac%7Be%5E%7Bz_i%7D%7D%7B%5Csum_%7Bj%3D1%7D%5EK%20e%5E%7Bz_j%7D%7D)

## Deployment

The model is available in the `CureMyLeaf.ipynb` file. At the end of the notebook, you'll find an executable cell to test the model using sample images provided in the `/WebImagesTest` directory.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/CrisHzz/CureMyLeaf
   cd cure-my-leaf
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the Cure My Leaf model:

   1. Open the `CureMyLeaf.ipynb` notebook in Jupyter or Google Colab.
   2. Run all cells up to the testing section.
   3. In the testing cell, provide the relative path to your leaf image:
      ```python
      image_path = 'WebImagesTest/your_image.jpg'
      predict_image(image_path)
      ```
   4. The model will output the predicted health status of the leaf.

For any questions or issues, please open an issue in the GitHub repository.


## Deploy web

Please enter to https://cure-my-leaf.vercel.app/ to try the model in web

