from data_load import load_and_prepare_data
import numpy as np

# Linear Discriminant Analysis (LDA)
class LDA:
    def __init__(self):
        self.priors_ = None
        self.mean_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def class_means(self, X, y):
        """Calculate means for each class."""
        return np.array([np.mean(X[y == c], axis=0) for c in np.unique(y)])

    def _class_cov(self, X, y):
        """Calculate the shared covariance matrix."""
        cov = np.zeros((X.shape[1], X.shape[1]))
        for i, c in enumerate(np.unique(y)):
            diff = X[y == c] - self.mean_[i]
            cov += self.priors_[i] * np.dot(diff.T, diff)
        return cov

    def fit(self, X, y):
        """Fit the LDA model."""
        self.classes_ = np.unique(y)
        self.mean_ = self.class_means(X, y)
        self.priors_ = np.bincount(y) / len(y)
        self.covariance_ = self._class_cov(X, y)
        self.coef_ = np.linalg.solve(self.covariance_, self.mean_.T).T
        self.intercept_ = -0.5 * np.diag(np.dot(self.mean_, self.coef_.T)) + np.log(self.priors_)
        return self

    def predict(self, X):
        """Predict using the LDA model."""
        discriminant_values = np.array([np.dot(X, self.coef_[c]) + self.intercept_[c] for c in range(len(self.classes_))])
        return np.argmax(discriminant_values, axis=0)


# Quadratic Discriminant Analysis (QDA)
class QDA:
    def __init__(self, regularization=1e-6):
        self.class_mean = None
        self.class_covariance = None
        self.class_inverse = None
        self.class_determinate = None
        self.regularization = regularization

    def fit(self, X_train, y_train):
        """Fit the QDA model."""
        classes = np.unique(y_train)
        num_classes = len(classes)
        num_features = X_train.shape[1]

        self.class_mean = np.zeros((num_classes, num_features))
        self.class_covariance = np.zeros((num_classes, num_features, num_features))
        self.class_inverse = []
        self.class_determinate = []

        for i, c in enumerate(classes):
            X_c = X_train[y_train == c]
            self.class_mean[i] = np.mean(X_c, axis=0)
            cov_c = np.cov(X_c, rowvar=False) + np.eye(num_features) * self.regularization
            self.class_covariance[i] = cov_c
            self.class_inverse.append(np.linalg.inv(cov_c))
            self.class_determinate.append(np.linalg.slogdet(cov_c)[1])

    def predict(self, X_test):
        """Predict using the QDA model."""
        if self.class_mean is None or self.class_covariance is None:
            raise RuntimeError("Model has not been trained yet. Please call the fit method first.")

        num_test_samples = X_test.shape[0]
        num_classes = len(self.class_mean)
        log_likelihoods = np.zeros((num_test_samples, num_classes))

        for i in range(num_classes):
            mean_diff = X_test - self.class_mean[i]
            inv_cov = self.class_inverse[i]
            exponent = np.sum(np.dot(mean_diff, inv_cov) * mean_diff, axis=1)
            log_likelihoods[:, i] = -0.5 * (self.class_determinate[i] + exponent)

        return np.argmax(log_likelihoods, axis=1)


# Gaussian Naive Bayes (GNB)
class GNB:
    def __init__(self):
        self.class_ = None
        self.class_prior = None
        self.class_mean = None
        self.class_variance = None

    def fit(self, X, y):
        """Fit the GNB model."""
        self.class_ = np.unique(y)
        self.class_prior = np.array([np.mean(y == c) for c in self.class_])
        self.class_mean = [np.mean(X[y == c], axis=0) for c in self.class_]
        self.class_variance = [np.var(X[y == c], axis=0) for c in self.class_]

    def predict(self, X):
        """Predict using the GNB model."""
        predictions = []

        for x in X:
            posteriors = []
            for i, c in enumerate(self.class_):
                prior = np.log(self.class_prior[i])
                likelihood = np.sum(np.log(np.exp(-((x - self.class_mean[i]) ** 2) / (2 * self.class_variance[i])) /
                                           np.sqrt(2 * np.pi * self.class_variance[i])))
                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.class_[np.argmax(posteriors)])

        return np.array(predictions)
