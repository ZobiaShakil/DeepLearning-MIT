#  MNIST Classification – Lab 2 Part 1

This project implements handwritten digit classification using the MNIST dataset. Two models are trained and compared to evaluate performance and generalization.


##  Models

### 1. Fully Connected Neural Network (FCN)
- **Input**: Flattened 28×28 images (784 features)
- **Hidden Layer**: 128 neurons, ReLU activation
- **Output**: 10 classes (digits 0–9)
- **Accuracy**: ~96–97%

### 2. Convolutional Neural Network (CNN)
- **Architecture**:  
  `Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling → Flatten → Fully Connected → Output`
- **Accuracy**: ~97%
- **Advantage**: Preserves spatial features for better generalization

---

##  Dataset

- **Source**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Format**: Grayscale digits (0–9), size 28×28 pixels

---

## Tools & Libraries

- Python
- PyTorch
- Torchvision
- Matplotlib

---

## Results
| Model | Accuracy | Notes |
|-------|----------|-------|
| FCN   | ~96–97%  | Good baseline performance |
| CNN   | ~97%     | More efficient and generalizes better |

---

# MIT 6.S191 - Lab 2(part2): Computer Vision - Debiasing Facial Detection

This project implements a **Debiasing Variational Autoencoder (DB-VAE)** to mitigate algorithmic bias in a facial detection system. Trained on the CelebA dataset, the model aims to achieve high and equitable accuracy across diverse demographics, addressing the critical issue of bias in AI.

##  Project Overview

The goal is to build a facial detection classifier that performs equally well across different skin tones and genders. A standard CNN trained on the CelebA dataset (which has inherent biases) shows uneven performance. This project uses a novel DB-VAE approach to learn the underlying data distribution and adaptively re-sample the training set to reduce this bias, all without needing manual annotation of sensitive attributes.

##  The Problem of Bias

Machine learning models can inherit and amplify biases present in their training data. The CelebA dataset is heavily skewed, with a majority of images being light-skinned females. A standard CNN trained on this data will naturally become better at recognizing faces with these over-represented features, leading to **algorithmic bias** against under-represented groups (e.g., darker-skinned individuals).

## The Solution: DB-VAE

The Debiasing Variational Autoencoder (DB-VAE) is a hybrid model that combines:
1.  **A Classifier:** To predict "face" or "not face".
2.  **A Variational Autoencoder (VAE):** To learn the **latent space** of human faces in an entirely unsupervised way.

### How DB-VAE Works:
1.  **Learns Latent Features:** The VAE encoder learns to compress images into a latent distribution (mean and variance), capturing core features like skin tone, pose, and accessories without any labels.
2.  **Adaptive Re-sampling:** During training, the model calculates how common each learned latent feature is in the dataset. It then **increases the sampling probability** for images with rare features (e.g., dark skin, hats) and decreases it for common ones.
3.  **Trains a Fairer Model:** By seeing rare examples more frequently, the final classifier becomes more robust and accurate across all demographics.

##  Results & Evaluation

The model was evaluated on a balanced test set from the PPB (Pilots Parliamentary Benchmark) dataset, categorized into four groups:
-   Light Female
-   Light Male
-   Dark Female
-   Dark Male

The DB-VAE's performance was compared against a standard CNN baseline.

**Key Result:** The DB-VAE model showed **more uniform accuracy across all demographics** compared to the standard CNN, which exhibited significant bias, performing worse on darker-skinned groups.

##  Technical Implementation

### Model Architecture:
-   **Encoder:** A CNN that outputs a class logit and parameters (mean & log-variance) for the latent distribution.
-   **Decoder:** A transposed convolutional network that reconstructs images from the latent space.
-   **Loss Function:** A combination of:
    -   **Reconstruction Loss:** L1 loss between input and reconstructed image.
    -   **Latent Loss:** KL Divergence to encourage a structured latent space.
    -   **Classification Loss:** Binary cross-entropy for face detection.

### Key Techniques:
-   **Reparameterization Trick:** Allows gradients to flow through the stochastic sampling operation.
-   **Adaptive Re-sampling:** The core debiasing mechanism, implemented in `get_training_sample_probabilities`.

**Course:** MIT 6.S191 - Introduction to Deep Learning
