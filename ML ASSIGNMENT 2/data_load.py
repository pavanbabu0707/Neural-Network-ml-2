import numpy as np
import pickle
import os

def load_and_prepare_data(root_path='E:/ML ASSIGNMENT 2/cifar10_data', as_grayscale=False):
    """Load raw CIFAR-10 data using pickle and return processed train and test sets."""
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'

    train_data, train_labels = [], []

    # Load the training data
    for batch_file in train_batches:
        batch_path = os.path.join(root_path, batch_file)

        if not os.path.exists(batch_path):
            print(f"File {batch_file} not found. Please ensure the dataset is correctly downloaded and placed in {root_path}.")
            return

        with open(batch_path, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
            train_data.append(batch[b'data'])
            train_labels.append(batch[b'labels'])

    # Concatenate training data and labels from all batches
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Load the test data
    test_path = os.path.join(root_path, test_batch)
    if not os.path.exists(test_path):
        print(f"File {test_batch} not found. Please ensure the dataset is correctly downloaded and placed in {root_path}.")
        return

    with open(test_path, 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
        test_data = test_batch[b'data']
        test_labels = test_batch[b'labels']

    # Reshape the data into proper image format (N, 32, 32, 3)
    train_data = train_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_data = test_data.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    # Convert to grayscale if specified
    if as_grayscale:
        train_data = np.dot(train_data[..., :3], [0.299, 0.587, 0.114])
        test_data = np.dot(test_data[..., :3], [0.299, 0.587, 0.114])

    return train_data, train_labels, test_data, test_labels
