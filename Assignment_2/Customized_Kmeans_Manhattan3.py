import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Read data from data.txt
data = pd.read_csv('TwoFeatures.csv')
# labels = data[:, 0]
# points = data[:, 1:]

x1 = data['x1']
x2 = data['x2']
X = np.array(list(zip(x1, x2)))
 

# Function to calculate Manhattan distance
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))
 
# Function to calculate the centroid of a cluster
def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)
 
# Function to calculate the distance between a point and a centroid
def calculate_distance(point, centroid):
    return manhattan_distance(point, centroid)
 
# Function to assign points to clusters
def assign_clusters(points, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in points:
        distances = [calculate_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
    return clusters
 
# Function to calculate precision, recall and F-score
# def calculate_metrics(clusters, labels):
#     precision = 0
#     recall = 0
#     f_score = 0
#     for cluster in clusters:
#         cluster_labels = [labels[np.where(np.all(points == point, axis=1))[0][0]] for point in cluster]
#         cluster_labels = set(cluster_labels)
#         tp = len(cluster_labels)
#         fp = len(cluster) - tp
#         fn = len(cluster_labels) - tp
#         if tp == 0:
#             precision = 0
#             recall = 0
#             f_score = 0
#         else:
#             precision = tp / (tp + fp)
#             recall = tp / (tp + fn)
#             f_score = 2 * (precision * recall) / (precision + recall)
#     return precision, recall, f_score
 
# Function to implement k-means clustering
def k_means(points, k):
    # Initialize k centroids
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    clusters = assign_clusters(points, centroids)
    # Iterate until clusters don't change
    while True:
        new_centroids = [calculate_centroid(cluster) for cluster in clusters]
        new_clusters = assign_clusters(points, new_centroids)
        if np.array_equal(clusters, new_clusters):
            break
        else:
            clusters = new_clusters
    # precision, recall, f_score = calculate_metrics(clusters, labels)
    # return precision, recall, f_score
    return centroids,clusters
 
# Vary the value of k from 1 to 10
k_values = range(1, 9)



for k in k_values:
    centroids,clusters = k_means(X,k)
    print(centroids)
    
