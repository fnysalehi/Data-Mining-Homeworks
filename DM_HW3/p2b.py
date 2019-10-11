from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import metrics
from sklearn import svm

store_data = pd.read_csv('svmdata2.csv')

def stepwise_kpca(store_data, gamma, n_components):
    """
    Implementation of a RBF kernel PCA.

    Arguments:
        X: A MxN dataset as NumPy array where the samples are stored as rows (M),
           and the attributes defined as columns (N).
        gamma: A free parameter (coefficient) for the RBF kernel.
        n_components: The number of components to be returned.

    """
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(store_data, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(K)

    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_pc

# X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
X = store_data.iloc[:,:-1].values
y = store_data.iloc[:, 2].values

# print("X:")
# print(X)
# print("y:")
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# print("X_train:")
# print(X_train)
# print("X_test:")
# print(X_test)
# print("y_train:")
# print(y_train)
# print("y_test:")
# print(y_test)


plt.figure(figsize=(8,6))

plt.scatter(X_train[:, 0], X_train[:,1], s=100, c=y_train, alpha=0.5)

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(store_data)

X_pc = stepwise_kpca(store_data, gamma=15, n_components=1)

plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0],y_train, c=y_train, alpha=0.5)
plt.scatter(X_train[:, 1], y_train, c=y_train, alpha=0.5)
plt.title('First principal component after RBF Kernel PCA')

plt.show()


#c_______________

clf = svm.SVC(kernel='linear') # Linear Kernel
X_spca = scikit_pca.fit_transform(store_data)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy_transformed_data:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision_transformed_data:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall_transformed_data:",metrics.recall_score(y_test, y_pred))

#D_________________________________

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

