BENCHMARK.PY 
The CIFAR-10 dataset is a widely used benchmark in machine learning and computer vision 
research. It consists of 60,000 32x32 color images across 10 classes, with 6,000 images 
per class. The dataset is split into 50,000 training images and 10,000 test images, providing 
a robust foundation for developing and evaluating image classification algorithms. 
The data set of cifar-10 has been loaded for all the three models and 
the code was used from benchmark.py provided in the assignment 

DATA LOAD: 
This code defines a function `load_and_prepare_data` to load and process 
the CIFAR-10 dataset. It reads the data from pickle files, concatenates 
multiple training batches, and reshapes the data into proper image format. 
The function handles both color and grayscale versions of the dataset, with an 
option to convert to grayscale. It includes error checking for file existence and 
returns the processed training and test data along with their corresponding 
labels. 
Using the above code to import the data to each model analysis. 

DISCRIMANT ANALYSIS: 
The provided code implements three classification algorithms: Linear 
Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and 
Gaussian Naive Bayes (GNB). Each class includes methods for fitting the 
model to training data and making predictions. LDA calculates shared 
covariance, while QDA allows for separate covariances per class, and GNB 
assumes feature independence. The implementations utilize efficient numpy 
operations, ensuring good performance for the CIFAR-10 dataset. 

Linear Discriminant Analysis: 
LDA is a dimensionality reduction technique that also serves as a classifier. It 
assumes a shared covariance matrix across all classes and estimates class 
means to make predictions. LDA is particularly effective when the classes are 
well-separated and have similar covariance structures. 
Linear Discriminant Analysis test accuracy on RGB Dataset:  
0.3713 
Linear Discriminant Analysis test accuracy on Grayscale Dataset: 
0.2739 

Quadratic Discriminant Analysis: 
QDA is an extension of LDA that allows for different covariance matrices for 
each class. This flexibility can capture more complex decision boundaries 
between classes, potentially leading to improved classification performance 
when classes have distinct covariance structures. 
Quadratic Discriminant Analysis test accuracy on RGB dataset: 
0.3623 
Quadratic Discriminant Analysis test accuracy on Grayscale dataset: 
0.4428 

Guardian Naïve Bayes Discriminant Analysis: 
Gaussian Naive Bayes is a probabilistic classifier that assumes features are 
independent given the class label. It models the distribution of each feature 
as a Gaussian and uses Bayes' theorem to make predictions. This method is 
computationally efficient and can perform well on various datasets. 
Guardian Naïve Bayes Discriminant Analysis test accuracy on RGB 
Dataset: 
0.2976 
Guardian Naïve Bayes Discriminant Analysis test accuracy on 
Grayscale Dataset: 
0.2662 

Conclusion: 
In conclusion, these three classifiers offer different approaches to tackling the 
CIFAR-10 image classification task. LDA and QDA provide discriminative 
methods with varying assumptions about class covariances, while Gaussian 
Naive Bayes offers a probabilistic perspective. Evaluating these models on 
both RGB and grayscale versions of the dataset will provide insights into their 
performance and the impact of color information on classification accuracy. 
The provided code implements three classification algorithms: Linear 
Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and 
Gaussian Naive Bayes (GNB). Each class includes methods for fitting the 
model to training data and making predictions. LDA calculates shared 
covariance, while QDA allows for separate covariances per class, and GNB 
assumes feature independence. The implementations utilize efficient numpy 
operations, ensuring good performance for the CIFAR-10 dataset. 
This code defines a function `load_and_prepare_data` to load and process 
the CIFAR-10 dataset. It reads the data from pickle files, concatenates 
multiple training batches, and reshapes the data into proper image format. 
The function handles both color and grayscale versions of the dataset, with an 
option to convert to grayscale. It includes error checking for file existence and 
returns the processed training and test data along with their corresponding 
labels. 
