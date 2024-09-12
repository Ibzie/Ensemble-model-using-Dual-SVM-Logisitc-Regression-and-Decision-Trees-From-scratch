import numpy as np
import pandas as pd

class KernelSVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=1.0):
        '''
        :param C: Regularization parameter (default: 1.0)
        :param kernel: Kernel type ('linear', 'polynomial', 'rbf')
        :param degree: Degree for polynomial kernel (default: 3)
        :param gamma: Gamma parameter for RBF kernel (default: 1.0)
        :return: None
        :Help with model taken from https://mi.eng.cam.ac.uk/~mjfg/local/4F10/lect9_pres.2up.pdf
    '''
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
    
    # Kernel functions for linear/nonlinear conditions
    def linear_kernel(self, X, Y):
        return np.dot(X, Y.T)
    
    def polynomial_kernel(self, X, Y):
        return (1 + np.dot(X, Y.T)) ** self.degree
    
    def rbf_kernel(self, X, Y):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        if len(Y.shape) == 1:
            Y = Y[np.newaxis, :]
        return np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - Y, axis=2) ** 2)
    
    def compute_kernel(self, X, Y):
        if self.kernel == 'linear':
            return self.linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, Y)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Kernel matrix
        K = self.compute_kernel(X, X)
        
        # Initialize Alpha (Which is the Lagrange Multiplier fi(alpha) thingy)
        alpha = np.zeros(n_samples)
        
        # Bias
        b = 0
        
        # Tolerance and learning rate
        tol = 1e-5
        lr = 1e-3
        
        y = np.where(y == 0, -1, 1)
        
        # Maximizing using Gradient Ascent
        for _ in range(1000):  # Can modify iterations for convergence
            for i in range(n_samples):
                # Calculate gradient for alpha_i
                gradient = 1 - y[i] * (np.sum(alpha * y * K[:, i]) + b)
                
                if (alpha[i] < self.C and gradient > tol) or (alpha[i] > 0 and gradient < -tol):
                    alpha[i] += lr * gradient
                    alpha[i] = np.clip(alpha[i], 0, self.C)  # Project alpha to [0, C]
            
            # Update the bias term
            b = np.mean(y - np.sum((alpha * y)[:, None] * K, axis=0))
        
        # Support vectors are where alpha > 0
        support_vector_indices = np.where(alpha > tol)[0]
        self.alpha = alpha[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.b = b
    
    # Prediction
    def predict(self, X):
        K = self.compute_kernel(X, self.support_vectors)
        return np.sign(np.sum(self.alpha * self.support_vector_labels * K, axis=1) + self.b)


# Synthetic Data for Linear Case
def generate_linear_data(n_samples=100):
    np.random.seed(1)
    X1 = np.random.randn(n_samples // 2, 2) - 1
    X2 = np.random.randn(n_samples // 2, 2) + 1
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    return X, y

# Synthetic Data for Nonlinear Case (Circle pattern)
def generate_nonlinear_data(n_samples=100):
    np.random.seed(1)
    X = np.random.randn(n_samples, 2)
    y = np.array(np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) < 1, dtype=int)
    return X, y

def cross_validation(model, X, y, k=5):
    X = np.array(X)
    y = np.array(y)
    n = len(X)
    fold_size = n // k
    for fold in range(k):
        X_train = np.concatenate([X[:fold*fold_size], X[(fold+1)*fold_size:]])
        y_train = np.concatenate([y[:fold*fold_size], y[(fold+1)*fold_size:]])
        X_test = X[fold*fold_size:(fold+1)*fold_size]
        y_test = y[fold*fold_size:(fold+1)*fold_size]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = np.mean(predictions == y_test)
        print(f"Fold {fold+1}, Accuracy: {accuracy:.4f}")

# # Generate linear dataset and fit the SVM
# X_linear, y_linear = generate_linear_data()
# svm_linear = KernelSVM(C=1.0, kernel='linear')
# svm_linear.fit(X_linear, y_linear)
# preds_linear = svm_linear.predict(X_linear)

# # Generate nonlinear dataset and fit the SVM with RBF kernel
# X_nonlinear, y_nonlinear = generate_nonlinear_data()
# svm_rbf = KernelSVM(C=1.0, kernel='rbf', gamma=0.5)
# svm_rbf.fit(X_nonlinear, y_nonlinear)
# preds_rbf = svm_rbf.predict(X_nonlinear)

# print(f"Linear SVM predictions: {preds_linear}")
# print(f"RBF Kernel SVM predictions: {preds_rbf}")
