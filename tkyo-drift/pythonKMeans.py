# This is good for vectors/matrices
import numpy as np

# This is the kmeans package
from sklearn.cluster import KMeans

# Allows the use of time functions
import time


def kMeansClustering(embeddings):

    # Starts the total function timer
    startTotal = time.perf_counter()

    # TODO Check if we need these error handling
    # Replace NaNs with zero
    embeddings = np.nan_to_num(embeddings)
    # Prevent extreme values
    embeddings = np.clip(embeddings, -1e6, 1e6)

    # Sets num_vectors to the amount of vectors
    num_vectors = len(embeddings)
    # Sets dims to the length of the first vector in the array
    dims = len(embeddings[0])

    # TODO Add section to readme about this
    # Determines the amount of clusters
    num_of_clusters = int(np.sqrt(num_vectors / 2))

    # Initialize the KMeans clustering algorithm with specific parameters:
    # random_state: Seed for random number generator to ensure reproducibility
    # (Using the same random_state will produce identical results across runs)
    kmeans = KMeans(num_of_clusters, random_state=42069)

    # Train the K-means model on our data
    # This is where the actual clustering algorithm runs:
    # 1. Initializes centroids (using k-means++ by default)
    # 2. Iteratively assigns points to nearest centroids
    # 3. Recomputes centroids as mean of assigned points
    # 4. Repeats until convergence or max iterations reached
    kmeans.fit(embeddings)

    # assigns the resulting array of centroids to centroids variable
    centroids = kmeans.cluster_centers_

    # TODO Remove prints before going live
    # This is printing the centroids to the console
    # The f is just for formatting
    print(f"Centroids:", centroids)

    # Print the shape of the centroids array to verify it matches expectations
    # Shape will be (n_clusters, dims)
    print(f"Centroids shape: {centroids.shape}")

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Elapsed: {endTotal - startTotal:.6f} seconds")

    return centroids
