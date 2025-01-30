from data_load import load_and_prepare_data
import numpy as np
from discrimant_analysis import QDA  # Ensure QDA is defined in your models module
from sklearn.metrics import accuracy_score

def flatten_images(X):
    """Flatten the images for model input."""
    return X.reshape(X.shape[0], -1)

def evaluate_qda_model(X_train, y_train, X_test, y_test):
    """Fit the QDA model and evaluate accuracy."""
    qda_model = QDA()  # Initialize QDA model
    qda_model.fit(X_train, y_train)  # Fit the model
    y_pred = qda_model.predict(X_test)  # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    return accuracy

# Load dataset
X_train_r, y_train_r, X_test_r, y_test_r = load_and_prepare_data(as_grayscale=False)
X_train_g, y_train_g, X_test_g, y_test_g = load_and_prepare_data(as_grayscale=True)

# Flatten the images
X_train_r_flat = flatten_images(X_train_r)
X_test_r_flat = flatten_images(X_test_r)
X_train_g_flat = flatten_images(X_train_g)
X_test_g_flat = flatten_images(X_test_g)

# Evaluate QDA model on RGB data
accuracy_rgb = evaluate_qda_model(X_train_r_flat, y_train_r, X_test_r_flat, y_test_r)
print("QDA accuracy on RGB test data:", accuracy_rgb)

# Evaluate QDA model on Grayscale data
accuracy_gray = evaluate_qda_model(X_train_g_flat, y_train_g, X_test_g_flat, y_test_g)
print("QDA accuracy on Grayscale test data:", accuracy_gray)
