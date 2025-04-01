# This is good for vectors/matrices
import numpy as np

# This is the hnsw package
import hnswlib

# This is the kmeans package
from sklearn.cluster import KMeans

# Allows the use of time functions
import time

# Define a reusable function to perform clustering and return centroids
def get_cluster_centroids(num_vectors, dims):

    # Define the models
    MODELS = {
        "semantic": "Xenova/all-MiniLM-L12-v2",
        # "concept": "Xenova/e5-base",
        "lexical": "Xenova/all-MiniLM-L6-v2",
    }

    # Define input and output types
    IO_TYPES = ["input", "output"]

    # Starts the total function timer
    startTotal = time.perf_counter()

    # Starts the Kmeans timer
    startKmeans = time.perf_counter()

    for models in MODELS:
        IO_LENGTH = len(IO_TYPES)
        for i in range(IO_LENGTH):
            trainingData = np.fromfile(
                f"data/{models}.{IO_TYPES[0]}.training.bin", dtype=np.float32
            )

            # Replace NaNs with zero
            trainingData = np.nan_to_num(trainingData)
            # Prevent extreme values
            trainingData = np.clip(trainingData, -1e6, 1e6)

            # Check that the length of the imported training data
            # This should
            if len(trainingData) == num_vectors * dims:
                trainingData = np.nan_to_num(trainingData)
                trainingData = trainingData.reshape((num_vectors, dims))
                # print("Successfully reshaped data:", trainingData.shape)
                # print("First vector:", trainingData[:5])
                # print(trainingData[0])
                # print(num_vectors * dims)
            else:
                raise ValueError(
                    f"Expected {num_vectors * dims} elements, "
                    f"but got {len(trainingData)}. Check file or dimensions."
                )

            # Sets dims to the length of the first vector in the array
            dims = len(trainingData[0])

            print("The value of dims is", dims)

            # Initialize the KMeans clustering algorithm with specific parameters:
            # n_clusters defines the amount of clusters to be created, aka the 'k' value
            # random_state=42: Seed for random number generator to ensure reproducibility
            # (Using the same random_state will produce identical results across runs)
            kmeans = KMeans(n_clusters=1000, random_state=72)

            # Train the K-means model on our data
            # This is where the actual clustering algorithm runs:
            # 1. Initializes centroids (using k-means++ by default)
            # 2. Iteratively assigns points to nearest centroids
            # 3. Recomputes centroids as mean of assigned points
            # 4. Repeats until convergence or max iterations reached
            kmeans.fit(trainingData)

            # assigns the resulting array of centroids to centroids variable
            centroids = kmeans.cluster_centers_

            # This is printing the centroids to the console
            # The f is just for formatting
            print(f"Centroids:", centroids)

            # Print the shape of the centroids array to verify it matches expectations
            # Shape will be (n_clusters, n_features) - in this case (1000, 700)
            print(f"Centroids shape: {centroids.shape}")

            endKmeans = time.perf_counter()
            print(f"Elapsed: {endKmeans - startKmeans:.6f} seconds")

            # Save all centroids to a binary .npy file
            # np.save('1000_clusters_centroids.npy', centroids)

            # Save centroids to a CSV file
            # np.savetxt('1000_clusters_centroids.csv', centroids, delimiter=',')

            # centroids.tofile('1000_clusters_centroids.bin')

            startHNSW = time.perf_counter()
            # Initialize HNSW index
            # 'l2' = Euclidean distance
            index = hnswlib.Index(space="l2", dim=dims)

            # Build the index
            # ef_construction : Controls build speed/accuracy trade-off
            # M:  Number of bidirectional links
            index.init_index(max_elements=len(trainingData), ef_construction=200, M=16)

            # Add data to the index
            index.add_items(centroids)

            # Save the index to disk
            index.save_index("hnsw_index.bin")

            # Number of neighbors to find
            k = 5
            # A random embedded query
            query = np.random.rand(dims).astype("float32")
            # Destructering labels and distances from the nearest neighbors query
            labels, distances = index.knn_query(query, k=k)

            # This will print the indices of the centroid for the nearest neighbors
            print(f"Nearest neighbors: {labels}")
            # These are the euclidian distances from the nearest neighbors to the centroid
            print(f"Distances: {distances}")

            # Ends timing for the HNSW

            endHNSW = time.perf_counter()
            print(f"Elapsed: {endHNSW - startHNSW:.6f} seconds")

            # Ends timing for the entire function
            endTotal = time.perf_counter()
            print(f"Elapsed: {endTotal - startTotal:.6f} seconds")

    # Exits the function
    return

# assigns the value of centroids to the evalued result of invoking the clusters function
centroids = get_cluster_centroids(num_vectors=25000, dims=384)
