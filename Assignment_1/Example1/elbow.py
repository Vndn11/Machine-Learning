# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Your data
# x = [2, 4, 2, 4, 6, 8, 6, 8, 4.5, 5.5, 5.5, 4.5]
# y = [11, 11, 9, 9, 11, 11, 9, 9, 5.5, 5.5, 4.5, 4.5]
# data = np.array(list(zip(x, y)))

# # Create a range of values for K, the number of clusters
# k_values = range(1, 10)

# # For each value of K, train the K-means algorithm on the data
# wss_values = []
# for k in k_values:
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(data)
#     wss_values.append(kmeans.inertia_)

# # Plot the WCSS values against K
# print(k_values,wss_values)
# plt.plot(k_values, wss_values)

# # Identify the point on the plot where the graph starts to flatten out
# elbow_point = plt.annotate("elbow", (2, wss_values[4]), fontsize=10)

# # The K value corresponding to the elbow of the curve is the optimal number of clusters
# optimal_k = 2  # In this example, we'll use K=2 as the optimal number of clusters based on the elbow method

# # Show the plot
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Your data
x = [2, 4, 2, 4, 6, 8, 6, 8, 4.5, 5.5, 5.5, 4.5]
y = [11, 11, 9, 9, 11, 11, 9, 9, 5.5, 5.5, 4.5, 4.5]
data = np.array(list(zip(x, y)))

# Create a range of values for K, the number of clusters
k_values = range(1, 10)

# Initialize an empty list to store the WCSS values
wcss_values = []

# Calculate the WCSS for each value of K
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss = kmeans.inertia_
    wcss_values.append(wcss)

# Plot the WCSS values against K
plt.plot(k_values, wcss_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')

# Identify the point on the plot where the graph starts to flatten out (the "elbow" point)
elbow_point = plt.annotate("elbow", (2, wcss_values[1]), fontsize=10, xytext=(-20, 20), textcoords='offset points',
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))

# The K value corresponding to the elbow of the curve is the optimal number of clusters
optimal_k = 2  # In this example, we'll use K=2 as the optimal number of clusters based on the elbow method

# Show the plot
plt.show()

# Print the WCSS values for all K values
for k, wcss in zip(k_values, wcss_values):
    print(f'K={k}, WCSS={wcss}')
