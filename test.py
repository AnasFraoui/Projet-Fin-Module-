import cv2
import joblib
import numpy as np
from skimage.feature import hog

# Load model
model = joblib.load("model.pkl")

# Load image
img_path = "dataset/cats/0.jpg"
img = cv2.imread(img_path, 0)
img = cv2.resize(img, (128,128))

# Extract HOG features
features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2))

features = np.array(features).reshape(1, -1)

# Prediction
prediction = model.predict(features)

if prediction[0] == 0:
    print("Cat 🐱 - test.py:23")
else:
    print("Dog 🐶 - test.py:25")