import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn import metrics

# Load the data
X = pd.read_csv('D:\CS 484 (Intro To ML)\Assignment 2\TwoFeatures.csv')
x1 = X['x1']
x2 = X['x2']
# X = np.array(list(zip(x1, x2)))

# a) Plot x2 versus x1
plt.scatter(x1, x2, marker='o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.title('Scatter Plot of x2 vs. x1')
plt.show()



def get_random_positions(num_observations, num_clusters):
    centroid_positions = []
    num_obs = 0
    sample_index = 0
    for obs_index in range(num_observations):
        num_obs += 1
        threshold = (num_clusters - sample_index) / (num_observations - num_obs + 1)
        if random.random() < threshold:
            centroid_positions.append(obs_index)
            sample_index += 1

        if sample_index == num_clusters:
            break

    return centroid_positions

def assign_members(X, centroids, distance_type):
    pair_distances = metrics.pairwise_distances(X, centroids, metric=distance_type)
    cluster_membership = pd.Series(np.argmin(pair_distances, axis=1), name='Cluster')
    wc_distances = pd.Series(np.min(pair_distances, axis=1), name='Distance')

    return cluster_membership, wc_distances

def k_means_cluster(X, num_clusters, distance_type='manhattan',num_iterations=500, num_trials=10, random_seed=None):
    num_observations = X.shape[0]

    if random_seed is not None:
        random.seed(a=random_seed)

    centroid_list = []
    twcv_list = []
    for trial_index in range(num_trials):
        centroid_positions = get_random_positions(num_observations, num_clusters)
        centroids = X.iloc[centroid_positions]
        
        cluster_membership_prev = pd.Series([-1] * num_observations, name='Cluster')
        
        for iteration in range(num_iterations):
            cluster_membership, wc_distances = assign_members(X, centroids, distance_type)

            centroids = X.join(cluster_membership).groupby(by=['Cluster']).mean()
            member_diff = np.sum(np.abs(cluster_membership - cluster_membership_prev))
            if member_diff > 0:
                cluster_membership_prev = cluster_membership
            else:
                break

        centroid_list.append(centroids)
        twcv_list.append(np.sum(np.power(wc_distances, 2)))
    best_solution = np.argmin(twcv_list)
    centroids = centroid_list[best_solution]

    cluster_membership, wc_distances = assign_members(X, centroids, distance_type)

    return cluster_membership, centroids, wc_distances

total_within_cluster_variance_list = []
elbow_list = []
num_clusters = range(1,9)
num_cluster_list = [1,2,3,4,5,6,7,8]
for k in num_clusters:
    cluster_membership, cluster_centroids, wc_distances = k_means_cluster(X, k)
    wc_distances_squared = np.square(wc_distances)
    total_within_cluster_variance = np.sum(wc_distances_squared)
    total_within_cluster_variance_list.append(total_within_cluster_variance)


    elbow = 0.0
    for i in range(k):
        wcv = np.sum(wc_distances_squared[cluster_membership == i])
        size = len(wc_distances[cluster_membership == i])
        elbow += wcv / size
    elbow_list.append(elbow)

Part2_Table = pd.DataFrame({
    "NO OF CLUSTERS": num_cluster_list,
    "TWCSS": total_within_cluster_variance_list,
    "ELBOW VALUE": elbow_list,
})

print(Part2_Table)

plt.plot(num_clusters, elbow_list, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow Value')
plt.title('Elbow Method for Optimal Cluster Selection')
plt.grid(True)
plt.show()

Slope=[0]* len(elbow_list)
Acceleration=[-1]* len(elbow_list)

for i in range(0,len(elbow_list)-1):
    if elbow_list[i+1] :
        Slope[i+1] = (elbow_list[i+1] - elbow_list[i])
        Acceleration[i+1] = (Slope[i+1] - Slope[i])

optimal_clusters = num_cluster_list[Acceleration.index(max(Acceleration)) - 1]
print(f"\nOptimal Number of Clusters (No Transformation): {optimal_clusters}")

a,Centroid,c = k_means_cluster(X, optimal_clusters)

print('\nCluster Centroids of Optimal Number of Cluster: \n',Centroid)


# c) Linearly rescale x1 and x2
x1_rescaled = (x1 - x1.min()) / (x1.max() - x1.min()) * 10
x2_rescaled = (x2 - x2.min()) / (x2.max() - x2.min()) * 10
X_rescaled = np.array(list(zip(x1_rescaled, x2_rescaled)))
X_rescaled = pd.DataFrame(X_rescaled)

plt.scatter(x1_rescaled, x2_rescaled, marker='o')
plt.xlabel('x1_rescaled')
plt.ylabel('x2_rescaled')
plt.grid(True)
plt.title('Scatter Plot of x2_rescaled vs. x1_rescaled')
plt.show()

total_within_cluster_variance_list = []
elbow_list = []
num_clusters = range(1,9)
num_cluster_list = [1,2,3,4,5,6,7,8]
for k in num_clusters:
    cluster_membership, cluster_centroids, wc_distances = k_means_cluster(X_rescaled, k)
    wc_distances_squared = np.square(wc_distances)
    total_within_cluster_variance = np.sum(wc_distances_squared)
    total_within_cluster_variance_list.append(total_within_cluster_variance)


    elbow = 0.0
    for i in range(k):
        wcv = np.sum(wc_distances_squared[cluster_membership == i])
        size = len(wc_distances[cluster_membership == i])
        elbow += wcv / size
    elbow_list.append(elbow)

Part3_Table = pd.DataFrame({
    "NO OF CLUSTERS": num_cluster_list,
    "TWCSS": total_within_cluster_variance_list,
    "ELBOW VALUE": elbow_list,
})

print(Part3_Table)

plt.plot(num_clusters, elbow_list, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow Value')
plt.title('Elbow Method for Optimal Cluster Selection (With Transformation)')
plt.grid(True)
plt.show()

Slope=[0]* len(elbow_list)
Acceleration=[-1]* len(elbow_list)

for i in range(0,len(elbow_list)-1):
    if elbow_list[i+1] :
        Slope[i+1] = (elbow_list[i+1] - elbow_list[i])
        Acceleration[i+1] = (Slope[i+1] - Slope[i])
# Determine the optimal number of clusters (Elbow point)
optimal_clusters = num_cluster_list[Acceleration.index(max(Acceleration)) - 1]
print(f"\nOptimal Number of Clusters (With Transformation): {optimal_clusters}")

a,Centroid,c = k_means_cluster(X_rescaled, optimal_clusters)

print('\nCluster Centroids of Optimal Number of Cluster(With Transformation): \n',Centroid)

