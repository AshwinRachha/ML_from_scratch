import numpy as np 
print(np.__version__)

# Subtract mean value from X
# Calculate COV(X, X) - covariance matrix
# Sort the eigenvectors according to their eigenvalues in decreasing order
# Choose the first k eigenvectors and that will be the new k dimensions
# Transform the original n dimensional data points into k dimensions

class PCA:

  def __init__(self, n_components):
    self.n_components = n_components
    self.components = None
    self.mean = None

  def fit(self, X):
    # mean
    self.mean = np.mean(X, axis = 0)
    X = X - self.mean
    # row = 1 sample, column = features
    # calculate cov matrix
    cov = np.cov(X.T)
    
    # eigenvectors and eigen values
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # sort eigenvectors
    eigenvectors = eigenvectors.T 
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]

    # store first n eigenvectors
    self.components = eigenvectors[:self.n_components]
  def transform(self, X):
    # project our data
    X = X - self.mean
    return np.dot(X, self.components.T)

if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use( 'tkagg' )
    from sklearn import datasets

    # data = datasets.load_digits()
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
