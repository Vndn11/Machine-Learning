import numpy as np
import pandas as pd
import random
import sys

# # Set some options for printing all the columns
# np.set_printoptions(precision=10, threshold=sys.maxsize)
# np.set_printoptions(linewidth=np.inf)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', None)

# pd.options.display.float_format = '{:,.10f}'.format

from sklearn import metrics

def get_random_positions(num_observations, num_clusters):
    centroid_positions = []
    num_obs = 0
    sample_index = 0
    for obs_index in range(num_observations):
        num_obs += 1
        threshold = (num_clusters - sample_index) / (num_observations - obs_index + 1)
        if random.random() < threshold:
            centroid_positions.append(obs_index)
            sample_index += 1

        if sample_index == num_clusters:
            break

    return centroid_positions

def assign_members(training_data, centroids, distance_type):
    pair_distances = metrics.pairwise_distances(training_data, centroids, metric=distance_type)
    cluster_membership = pd.Series(np.argmin(pair_distances, axis=1), name='Cluster')
    wc_distances = pd.Series(np.min(pair_distances, axis=1), name='Distance')

    return cluster_membership, wc_distances

def k_means_cluster(training_data, num_clusters, distance_type='manhattan',num_iterations=500, num_trials=10, random_seed=None):
    num_observations = training_data.shape[0]

    if random_seed is not None:
        random.seed(a=random_seed)

    centroid_list = []
    twcv_list = []
    for trial_index in range(num_trials):
        centroid_positions = get_random_positions(num_observations, num_clusters)
        centroids = training_data.iloc[centroid_positions]
        print('centroid_positions', centroid_positions)
        print('centroids', centroids)
        cluster_membership_prev = pd.Series([-1] * num_observations, name='Cluster')
        
        for iteration in range(num_iterations):
            cluster_membership, wc_distances = assign_members(training_data, centroids, distance_type)

            centroids = training_data.join(cluster_membership).groupby(by=['Cluster']).mean()
            member_diff = np.sum(np.abs(cluster_membership - cluster_membership_prev))
            if member_diff > 0:
                cluster_membership_prev = cluster_membership
            else:
                break

        centroid_list.append(centroids)
        twcv_list.append(np.sum(np.power(wc_distances, 2)))
    
    # Find the first choice with the smallest twcv
    best_solution = np.argmin(twcv_list)
    centroids = centroid_list[best_solution]

    cluster_membership, wc_distances = assign_members(training_data, centroids, distance_type)

    return cluster_membership, centroids, wc_distances

# training_data = pd.DataFrame({'x': [0.1, 0.3, 0.4, 0.8, 0.9]})
training_data = pd.read_csv('TwoFeatures.csv')


num_clusters = 8
cluster_membership, cluster_centroids, wc_distances = k_means_cluster(training_data, num_clusters)
wc_distances_squared = np.square(wc_distances)
total_within_cluster_variance = np.sum(wc_distances_squared)

elbow = 0.0
for i in range(num_clusters):
    wcv = np.sum(wc_distances_squared[cluster_membership == i])
    size = len(wc_distances[cluster_membership == i])
    elbow += wcv / size
