import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# pd.options.display.float_format = '{:,.10f}'.format
class CustomKMeansManhattan:
    def __init__(self, n_clusters=4, max_iters=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None

    def fit(self, Y):
        np.random.seed(self.random_state)
        
        # Randomly initialize cluster centroids
        self.centroids = Y[np.random.choice(Y.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_clusters(Y)
            
            # Update centroids based on the mean of data points in each cluster
            new_centroids = np.array([Y[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Calculate inertia (TWCSS)
            # twcss = np.sum([np.sum(np.abs(Y[labels == i] - new_centroids[i])) for i in range(self.n_clusters)])
            
            
            # print([np.sum(np.abs(X[labels == i] - new_centroids[i])) for i in range(self.n_clusters)])
            # print(np.sum([np.sum(np.square(np.abs(X[labels == i] - new_centroids[i]))) for i in range(self.n_clusters)]))
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                twcss = np.sum([np.sum(np.square(np.abs(Y[labels == i] - new_centroids[i]))) for i in range(self.n_clusters)])
                print(twcss)
                # print([str(i) + '   ' + str(((Y[labels == i] ))) for i in range(self.n_clusters)])
                break
            
            self.centroids = new_centroids

        # Store the calculated inertia
        self.inertia_ = twcss
    def _assign_clusters(self, Y):
        distances = np.sum(np.abs(Y[:, np.newaxis] - self.centroids), axis=2)
        return np.argmin(distances, axis=1)

# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate sample data
    # Y, _ = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)
    X = pd.read_csv('D:\CS 484 (Intro To ML)\Assignment 2\TwoFeatures.csv')
    # Y = X.to_numpy()
    x1 = X['x1']
    x2 = X['x2']
    Y = np.array(list(zip(x1, x2)))
    # print(Y[np.random.choice(Y.shape[0], 4, replace=False)])
    
    
    # Instantiate and fit the custom K-Means with Manhattan distance
    custom_kmeans = CustomKMeansManhattan(n_clusters=4)
    custom_kmeans.fit(Y)
    print(custom_kmeans.centroids)
    # Get cluster labels
    labels = custom_kmeans._assign_clusters(Y)
    inertia = custom_kmeans.inertia_
    
    # print(Y[np.random.choice(Y.shape[0], 1, replace=False)])
    


    # Plot the data points and centroids
    # plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis')
    # plt.scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    # plt.legend()
    # plt.title('Custom K-Means Clustering with Manhattan Distance')
    # plt.show()
    # print(Y[np.random.choice(Y.shape[0], 4, replace=False)])





    # centroids = [[1.2500,-1.8663],[-2.3500,1.3284],[-8.7000,-1.6550],[-1.8500,-2.3240]]
    
    # distances = np.sum(np.abs(Y[:, np.newaxis] - centroids), axis=2)
    # print(distances)
    # print(np.argmin(distances,axis=1))
    # kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
    # kmeans.fit(Y)
    # Centroid_x = kmeans.cluster_centers_[:, 0]
    # Centroid_y = kmeans.cluster_centers_[:, 1]
    # plt.scatter(Y[:, 0], Y[:, 1], c=kmeans.labels_, cmap='viridis')
    # plt.scatter(Centroid_x,Centroid_y, c='red', marker='x', s=100, label='Centroids')
    # plt.legend()
    # plt.title('Custom K-Means Clustering with Manhattan Distance')
    # plt.show()



