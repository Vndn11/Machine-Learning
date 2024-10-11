def kmeans(data, k, centroids):
  """
  Performs k-means clustering on the given data.

  Args:
    data: The data to cluster.
    k: The number of clusters.
    centroids: The initial centroids of the clusters.

  Returns:
    A list of cluster assignments.
  """

  # Initialize the cluster assignments.
  assignments = [None] * len(data)

  # Iterate until the assignments do not change.
  while True:
    # Update the cluster assignments.
    for i in range(len(data)):
      distances = [np.linalg.norm(data[i] - centroid) for centroid in centroids]
      min_index = np.argmin(distances)
      assignments[i] = min_index

    # Check if the assignments have changed.
    if np.array_equal(assignments, old_assignments):
      break

    # Update the centroids.
    for i in range(k):
      centroids[i] = np.mean(data[assignments == i], axis=0)

  return assignments

