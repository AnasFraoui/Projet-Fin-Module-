import os
import cv2
import numpy as np

from skimage.feature import hog

# Dataset path
dataset_path = "dataset"

X = []
y = []

categories = ["cats", "dogs"]

for label, category in enumerate(categories):

    folder_path = os.path.join(dataset_path, category)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        # Read image in grayscale
        img = cv2.imread(image_path, 0)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (128, 128))

        # Extract HOG features
        features = hog(
            img,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            visualize=False
        )

        X.append(features)
        y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("Images loaded : - main.py:47", len(X))
print("Labels loaded : - main.py:48", len(y))
print("Feature vector size : - main.py:49", X[0].shape)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create MLP model
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=42
)

# Train model
print("Training model... - main.py:70")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy : - main.py:79", accuracy * 100, "%")

import joblib

joblib.dump(model, "model.pkl")
print("Model saved! - main.py:84")
joblib.dump(model, "model.pkl")