# CIFAR-100 Image Classification using Custom CNN

This repository contains code for image classification on the CIFAR-100 dataset using a custom-built Convolutional Neural Network (CNN) architecture. It also includes comparisons with various pretrained models such as ResNet-50, VGG-19, DenseNet-121, and EfficientNetB0.

## 📂 Dataset

The CIFAR-100 dataset consists of 60,000 color images (32x32 pixels) in 100 classes, with 600 images per class:
- Training set: 50,000 images
- Test set: 10,000 images

Dataset Source: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🧪 Preprocessing

- Normalization of images to range [0, 1]
- One-hot encoding for labels
- Data augmentation:
  - Random horizontal flips
  - Random cropping
  - Brightness and contrast adjustments

## 🧠 Custom CNN: Architecture & Design Philosophy

This is a **from-scratch** convolutional neural network designed specifically for CIFAR-100...

### 🔧 Architecture Summary

```python
# Block 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Block 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Dense Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='softmax'))
```


## ⚙️ Training Configuration

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Learning Rate: 0.001
- Epochs: 100
- Batch Size: 128
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

## 📈 Results

| Model            | Train Accuracy | Val Accuracy | Test Accuracy |
|------------------|----------------|--------------|---------------|
| **Custom CNN**   | 57.3%          | 58.2%        | 56.6%         |
| ResNet-50        | 65.9%          | 68.7%        | 68.0%         |
| VGG-19           | 57.9%          | 63.1%        | 62.5%         |
| DenseNet-121     | 68.4%          | 68.2%        | 67.5%         |
| EfficientNetB0   | 74.4%          | 72.4%        | 72.3%         |

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Filter Visualization (Pre and Post-training)

## 💡 Key Takeaways

- Custom CNN demonstrates strong performance and is comparable to VGG-19.
- EfficientNetB0 achieves the highest test accuracy, showcasing the power of transfer learning.
- Regularization techniques such as Dropout and Batch Normalization effectively mitigate overfitting.

## 📎 References

- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Various CNN Architectures: ResNet, DenseNet, EfficientNet (referenced for performance comparison and benchmarking)

---

## 👤 Author

**Rachakonda Akhil Goud**  
Master's Student in Computer Science  
[GitHub Profile](https://github.com/akhilrachakonda)


