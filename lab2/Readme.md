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
