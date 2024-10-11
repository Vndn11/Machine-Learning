import numpy as np
import pandas as pd


def CustomKMeansManhattan(data,n_clusters):
    centroids = Y[np.random.choice(Y.shape[0], self.n_clusters, replace=False)]
    

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    Y = pd.read_csv('D:\CS 484 (Intro To ML)\Assignment 2\TwoFeatures.csv')
    # Y = X.to_numpy()
    x1 = Y['x1']
    x2 = Y['x2']
    X = np.array(list(zip(x1, x2)))
    
    k_range = range(1,9)

    for k in k_range:
        custom_kmeans = CustomKMeansManhattan(X,k)





    # Instantiate and fit the custom K-Means with Manhattan distance
    custom_kmeans = CustomKMeansManhattan(n_clusters=4)
    custom_kmeans.fit(X)
    
    # Get cluster labels and TWCSS (inertia)
    labels = custom_kmeans.labels
    inertia = custom_kmeans.inertia_
    
    print(f"Cluster Labels: {labels}")
    print(f"TWCSS (Inertia): {inertia}")
