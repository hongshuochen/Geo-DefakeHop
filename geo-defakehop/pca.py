import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as LA

class myPCA:
    def __init__(self) -> None:
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = None

    def fit(self, X):
        X = X.astype("float32")
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        C = X.T @ X
        C /= (len(X)-1)
        w, v = LA.eig(C)
        v = v.T
        w = np.abs(w)
        idx = np.argsort(w)[::-1]
        self.eigenvalues = w[idx]
        self.eigenvectors = v[idx]
        return self

    def fit_transform(self, X, n_components=-1):
        X = X.astype("float32")
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        C = X.T @ X
        C /= (len(X)-1)
        w, v = LA.eig(C)
        v = v.T
        w = np.abs(w)

        idx = np.argsort(w)[::-1]
        self.eigenvalues = w[idx]
        self.eigenvectors = v[idx]
        if n_components < 1 and n_components > 0:
            for i in range(1, len(self.eigenvalues)):
                if np.sum(self.eigenvalues[:i])/np.sum(self.eigenvalues) > n_components:
                    n_components = i
                    break

        if n_components == -1:
            return X @ self.eigenvectors.T
        else:
            return X @ self.eigenvectors[:n_components].T
        
    def transform(self, X, n_components=-1):
        X = X.astype("float32")
        X -= self.mean
        if n_components < 1 and n_components > 0:
            for i in range(1, len(self.eigenvalues)):
                if np.sum(self.eigenvalues[:i])/np.sum(self.eigenvalues) > n_components:
                    n_components = i
                    break

        if n_components == -1:
            return X @ self.eigenvectors.T
        else:
            return X @ self.eigenvectors[:n_components].T

    def inverse_transform(self, Y):
        Y = Y.astype("float32")
        return Y @ self.eigenvectors[:Y.shape[1]] + self.mean

if __name__ == "__main__":
    X = np.array([[-1.2, -1.5, 0], [-2, -1, 1], [-3, -2, 2], [1, 1, 3], [2, 1, 4], [3, 2, 5]])
    pca = PCA(n_components=3)
    pca.fit(X)
    mypca = myPCA()
    mypca.fit(X)

    print(pca.mean_)
    print(mypca.mean) 

    print(pca.explained_variance_)
    print(mypca.eigenvalues)
    print(np.var(pca.transform(X), axis=0, ddof=1))
    
    print(pca.explained_variance_ratio_)
    print(mypca.eigenvalues/np.sum(mypca.eigenvalues))
    
    print(pca.components_)
    print(mypca.eigenvectors)

    print(pca.transform(X))
    print(mypca.transform(X))

    print(X)
    print(pca.inverse_transform(pca.transform(X)))
    print(mypca.inverse_transform(mypca.transform(X)))
    
