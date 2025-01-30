from data_load import load_and_prepare_data
from discrimant_analysis import GNB  # Ensure GNB is defined in your models module
from sklearn.metrics import accuracy_score

def evaluate_gnb_model(X_train, y_train, X_test, y_test):
    """Fit the GNB model and evaluate accuracy."""
    gnb_model = GNB()  # Initialize GNB model
    gnb_model.fit(X_train, y_train)  # Fit the model
    y_pred = gnb_model.predict(X_test)  # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    return accuracy

# Load the dataset
X_train_r, y_train_r, X_test_r, y_test_r = load_and_prepare_data(as_grayscale=False)
X_train_g, y_train_g, X_test_g, y_test_g = load_and_prepare_data(as_grayscale=True)

# Evaluate GNB model on RGB data
accuracy_rgb = evaluate_gnb_model(X_train_r, y_train_r, X_test_r, y_test_r)
print("GNB accuracy on RGB test data:", accuracy_rgb)

# Evaluate GNB model on Grayscale data
accuracy_gray = evaluate_gnb_model(X_train_g, y_train_g, X_test_g, y_test_g)
print("GNB accuracy on Grayscale test data:", accuracy_gray)
