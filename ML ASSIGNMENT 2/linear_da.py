from data_load import load_and_prepare_data
from discrimant_analysis import LDA
from sklearn.metrics import accuracy_score

# Load data
root_path = 'E:/ML ASSIGNMENT 2/cifar10_data'
X_train_r, y_train_r, X_test_r, y_test_r = load_and_prepare_data(as_grayscale=False)
X_train_g, y_train_g, X_test_g, y_test_g = load_and_prepare_data(as_grayscale=True)

# Flatten the RGB and grayscale images
X_train_r_flat = X_train_r.reshape(X_train_r.shape[0], -1)
X_test_r_flat = X_test_r.reshape(X_test_r.shape[0], -1)
X_train_g_flat = X_train_g.reshape(X_train_g.shape[0], -1)
X_test_g_flat = X_test_g.reshape(X_test_g.shape[0], -1)

# Initialize LDA models for RGB and grayscale data
lda_rgb = LDA()
lda_gray = LDA()

# Fit the models to the training data
lda_rgb.fit(X_train_r_flat, y_train_r)
lda_gray.fit(X_train_g_flat, y_train_g)

# Make predictions on the test data
y_pred_rgb = lda_rgb.predict(X_test_r_flat)
y_pred_gray = lda_gray.predict(X_test_g_flat)

# Calculate and print accuracy scores
accuracy_rgb = accuracy_score(y_test_r, y_pred_rgb)
accuracy_gray = accuracy_score(y_test_g, y_pred_gray)

print(f"LDA accuracy on RGB test data: {accuracy_rgb:.4f}")
print(f"LDA accuracy on grayscale test data: {accuracy_gray:.4f}")
