

# Computer Vision – Handwritten Digit Classification with CNN

## 1. Task Description

The goal of this assignment is to build a **Convolutional Neural Network (CNN)** that can classify handwritten digit images into one of **10 classes**, representing the digits:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

This is a **supervised machine learning classification task**. The model receives an image as input and predicts which digit is shown.

The implementation uses **TensorFlow/Keras** to build and train a Convolutional Neural Network.

---

## 2. Pipeline Overview

```text
Load Data
    ↓
Separate Features and Labels
    ↓
One-Hot Encode Labels
    ↓
Normalize Pixel Values
    ↓
Reshape Data into Images
    ↓
Train/Test Split
    ↓
Build CNN
    ↓
Compile Model
    ↓
Train Model
    ↓
Evaluate Model
    ↓
Predict Test Data
    ↓
Create Submission File
```

---

## 3. Step 1: Import Libraries

```python
import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input

from sklearn.model_selection import train_test_split
```

### Explanation

| Library / Module   | Purpose                                                               |
| ------------------ | --------------------------------------------------------------------- |
| `pandas`           | Loading and processing the CSV datasets                               |
| `numpy`            | Numerical operations and processing predictions                       |
| `to_categorical`   | Converts class labels into one-hot encoded vectors                    |
| `Sequential`       | Creates a sequential neural network model                             |
| `Conv2D`           | Applies convolutional filters to image data                           |
| `MaxPooling2D`     | Reduces the spatial dimensions of feature maps                        |
| `Flatten`          | Converts multi-dimensional feature maps into a one-dimensional vector |
| `Dense`            | Creates fully connected neural network layers                         |
| `RMSprop`          | Optimizer used during training                                        |
| `train_test_split` | Splits the training data into training and validation data            |
| `Input`            | Defines the shape of the input images                                 |

---

## 4. Step 2: Load the Dataset

```python
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
```

### Explanation

Two datasets are loaded:

* `train.csv` contains the training examples and their labels.
* `test.csv` contains unlabeled images for which the model should generate predictions.

The training dataset contains a column called:

```text
label
```

This column represents the correct digit for each image.

The remaining columns contain the pixel values of the images.

---

## 5. Step 3: Separate Features and Labels

```python
X = train.drop(labels=['label'], axis=1)
y = train['label']
```

### Explanation

The dataset is separated into:

* `X`: input features, containing the pixel values of the images
* `y`: target labels, containing the correct digit

For example:

```text
Image pixels → 784 numerical values
Label        → digit between 0 and 9
```

The original image size is:

```text
28 × 28 pixels
```

Since:

```text
28 × 28 = 784
```

each image is initially represented by 784 pixel values in the CSV file.

---

## 6. Step 4: One-Hot Encode the Labels

```python
Y = to_categorical(y, num_classes=10)
```

### Explanation

The digit labels are converted into **one-hot encoded vectors**.

For example:

| Digit | One-Hot Encoding                 |
| ----- | -------------------------------- |
| 0     | `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |
| 1     | `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]` |
| 2     | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]` |

There are 10 possible output classes, so each label is represented by a vector of length 10.

One-hot encoding is used because the model uses:

```python
loss='categorical_crossentropy'
```

which expects the target classes in categorical form.

---

## 7. Step 5: Normalize Pixel Values

```python
X = X / 255.0
```

### Explanation

Pixel values are typically represented as integers between:

```text
0 and 255
```

The data is normalized to a range between:

```text
0 and 1
```

by dividing every pixel value by `255.0`.

For example:

```text
255 → 1.0
128 → 0.502
0   → 0.0
```

Normalization helps the neural network train more effectively because the input values are on a consistent scale.

---

## 8. Step 6: Reshape the Data into Images

```python
X = X.values.reshape(-1, 28, 28, 1)
```

### Explanation

The CSV dataset initially represents each image as a single row containing 784 pixel values.

The CNN expects image data with spatial dimensions, so the data is reshaped into:

```text
(number of images, height, width, channels)
```

For this dataset:

```text
(number of images, 28, 28, 1)
```

The final dimension `1` represents a single color channel because the images are grayscale.

---

## 9. Step 7: Build the Convolutional Neural Network

```python
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Model Architecture

```text
Input Image
28 × 28 × 1
        ↓
Conv2D
32 filters, 3 × 3
        ↓
ReLU
        ↓
Max Pooling
        ↓
Conv2D
64 filters, 3 × 3
        ↓
ReLU
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense Layer
512 neurons + ReLU
        ↓
Output Layer
10 neurons + Softmax
```

### Explanation of the Layers

#### Input Layer

```python
Input(shape=(28,28,1))
```

Defines the shape of one input image:

```text
28 × 28 pixels
1 color channel
```

---

#### Convolutional Layers

```python
Conv2D(32, (3,3), activation='relu')
```

and

```python
Conv2D(64, (3,3), activation='relu')
```

The convolutional layers apply filters to the image to detect visual patterns.

Early layers can learn simple patterns such as:

* edges
* lines
* curves

Later layers can combine these features into more complex representations useful for recognizing handwritten digits.

The first convolutional layer uses **32 filters**, while the second uses **64 filters**.

---

#### ReLU Activation

The convolutional and dense layers use the ReLU activation function:

```text
ReLU(x) = max(0, x)
```

ReLU introduces non-linearity into the network.

For example:

```text
ReLU(-3) = 0
ReLU(2) = 2
```

This allows the neural network to learn more complex relationships than a purely linear model.

---

#### Max Pooling

```python
MaxPooling2D(2, 2)
```

Max pooling reduces the spatial dimensions of the feature maps.

It keeps the strongest activation in each small region.

This helps:

* reduce the amount of data
* reduce computational complexity
* retain important visual features

---

#### Flatten Layer

```python
Flatten()
```

The convolutional layers produce multi-dimensional feature maps.

The `Flatten()` layer converts them into a single vector so they can be processed by the fully connected layers.

Conceptually:

```text
Feature Maps
      ↓
Flatten
      ↓
One-dimensional vector
```

---

#### Dense Hidden Layer

```python
Dense(512, activation='relu')
```

This is a fully connected neural network layer with 512 neurons.

It combines the features extracted by the convolutional layers to learn patterns useful for digit classification.

---

#### Output Layer

```python
Dense(10, activation='softmax')
```

The output layer contains 10 neurons, one for each digit:

```text
0 1 2 3 4 5 6 7 8 9
```

The `softmax` activation converts the output values into probabilities.

For example:

```text
Digit 0 → 0.01
Digit 1 → 0.02
Digit 2 → 0.91
...
```

The class with the highest probability is selected as the prediction.

---

## 10. Step 8: Split Training and Validation Data

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

### Explanation

The labeled dataset is split into:

* **80% training data**
* **20% validation/test data**

The model learns from:

```python
X_train
Y_train
```

and is evaluated using:

```python
X_test
Y_test
```

The parameter:

```python
random_state=42
```

ensures that the split is reproducible.

---

## 11. Step 9: Prepare the Unlabeled Test Data

```python
test = test / 255.0
test = test.values.reshape(-1, 28, 28, 1)
```

### Explanation

The unlabeled test dataset must receive the same preprocessing as the training data.

The pixel values are:

1. normalized to values between 0 and 1
2. reshaped into `28 × 28 × 1` images

This ensures that the model receives input data in the same format during prediction as during training.

---

## 12. Step 10: Compile the Model

```python
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Explanation

Before training, the model needs to be configured.

#### Optimizer: RMSprop

```python
optimizer=RMSprop()
```

The optimizer adjusts the model's weights during training to reduce the prediction error.

#### Loss Function: Categorical Cross-Entropy

```python
loss='categorical_crossentropy'
```

This loss function is suitable for multi-class classification when the target labels are one-hot encoded.

#### Metric: Accuracy

```python
metrics=['accuracy']
```

Accuracy measures the proportion of correctly classified images.

---

## 13. Step 11: Train the Model

```python
history = model.fit(
    X_train,
    Y_train,
    batch_size=100,
    epochs=5
)
```

### Explanation

The model is trained using the training data.

#### Batch Size

```python
batch_size=100
```

The model processes 100 training examples before updating its weights.

#### Epochs

```python
epochs=5
```

The model goes through the complete training dataset five times.

During each epoch, the model learns patterns that improve its ability to classify handwritten digits.

---

## 14. Step 12: Evaluate the Model

```python
loss, accuracy = model.evaluate(X_test, Y_test)

print("Test Accuracy:", accuracy)
```

### Explanation

After training, the model is evaluated using the held-out data.

The evaluation returns:

* **Loss:** how different the predictions are from the correct labels
* **Accuracy:** percentage of correctly classified images

### Result

Based on the recorded training output:

```text
Test Accuracy: 0.990476
```

This corresponds to approximately:

**99.05% accuracy**

This indicates that the CNN performed very well on the held-out validation data.

---

## 15. Step 13: Predict the Unlabeled Test Data

```python
results = model.predict(test)
```

### Explanation

The model predicts probabilities for each of the 10 digit classes.

For example, one prediction might look conceptually like:

```text
[0.00, 0.01, 0.02, 0.95, 0.01, ...]
```

The model does not directly return the digit label. Instead, it returns a probability for every possible class.

---

## 16. Step 14: Convert Predictions Back to Digit Labels

```python
results = np.argmax(results, axis=1)
```

### Explanation

`np.argmax()` selects the index with the highest predicted probability.

For example:

```text
[0.01, 0.02, 0.90, 0.03, ...]
```

The highest probability is at index `2`, so the predicted digit is:

```text
2
```

This converts the model's probability output back into a single class label.

---

## 17. Step 15: Create the Submission File

```python
results = pd.Series(results, name="Label")

submission = pd.concat([
    pd.Series(
        range(1, len(results)+1),
        name="ImageId"
    ),
    results
], axis=1)
```

### Explanation

A new DataFrame is created containing:

| ImageId | Label           |
| ------- | --------------- |
| 1       | predicted digit |
| 2       | predicted digit |
| 3       | predicted digit |

The `ImageId` identifies the image, while `Label` contains the predicted digit.

---

## 18. Step 16: Save the Results

```python
submission.to_csv(
    'data/submission.csv',
    index=False
)
```

### Explanation

The predictions are saved as a CSV file:

```text
submission.csv
```

The parameter:

```python
index=False
```

prevents pandas from adding an additional index column.

---

## 19. Results

The model achieved the following validation accuracy:

| Metric                   | Result                    |
| ------------------------ | ------------------------- |
| Validation/Test Accuracy | **99.05%**                |
| Epochs                   | 5                         |
| Batch Size               | 100                       |
| Optimizer                | RMSprop                   |
| Loss Function            | Categorical Cross-Entropy |

The high validation accuracy shows that the CNN successfully learned useful visual patterns for recognizing handwritten digits.

---

## 20. Key Concepts

| Concept             | Explanation                                                    |
| ------------------- | -------------------------------------------------------------- |
| Supervised Learning | Learning from labeled input-output examples                    |
| Classification      | Predicting one category from several possible classes          |
| CNN                 | Neural network architecture designed for image data            |
| Convolution         | Applies filters to detect visual patterns                      |
| Max Pooling         | Reduces feature map size while retaining important information |
| ReLU                | Activation function defined as `max(0, x)`                     |
| Softmax             | Converts output values into class probabilities                |
| One-Hot Encoding    | Represents categorical labels as binary vectors                |
| Normalization       | Scales numerical values to a consistent range                  |
| Epoch               | One complete pass through the training dataset                 |
| Batch Size          | Number of samples processed before updating model weights      |

---

## 21. Conclusion

This assignment demonstrates a complete **computer vision and deep learning pipeline** for handwritten digit classification.

The workflow includes:

* loading image data from CSV files
* separating features and labels
* one-hot encoding the target labels
* normalizing pixel values
* reshaping flat pixel data into image format
* building a Convolutional Neural Network
* training and evaluating the model
* generating predictions for unseen images
* creating a submission CSV file

The CNN achieved approximately **99.05% accuracy** on the held-out validation data after only five epochs.

This lab demonstrates why CNNs are particularly useful for image classification tasks: the convolutional layers can automatically learn relevant visual features, such as edges, shapes, and patterns, directly from the image data.

