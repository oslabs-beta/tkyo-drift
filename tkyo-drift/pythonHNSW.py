# Numerical operations package
import numpy as np
# HNSW nearest neighbor search package
import hnswlib
# Allows the use of time functions
import time
# Allows the use of file system operations
import os
# Allows the use of math functions
import math
# JSON serialization/deserialization
import json
# command-line argument handling
import sys

# TODO Check if we need to use HNSW for rolling data, if so modify code
# TODO Move file location to util

def HNSW(io_type, model_type, query):

    # Parse the JSON query string into a numpy array
    try:
        query = np.array(json.loads(query), dtype=np.float32)
    except json.JSONDecodeError:
        raise ValueError("Invalid query format - must be JSON string")

    def load_embeddings(filename):
        # Loads the embeddings from the binary file with the header
        with open(filename, "rb") as f:
            # Read and parse header containing num_vector and dims
            header = np.frombuffer(f.read(8), dtype=np.uint32)
            # Destructure num_vectors and dims from the headers
            num_vectors, dims = header[0], header[1]

            # Calculate expected data size (4 bytes per float32)
            expected_bytes = num_vectors * dims * 4

            # Read the exact amount of data
            data = np.frombuffer(f.read(expected_bytes), dtype=np.float32)

            # Check that the length of the imported training data
            if len(data) != num_vectors * dims:
                raise ValueError("Data is incorrect - size mismatch")

            # Returns the vectors from the binary file
            return data.reshape(num_vectors, dims), num_vectors, dims

    # Define file paths
    kmeans_file = f"data/{model_type}.{io_type}.kmeanstraining.bin"
    training_file = f"data/{model_type}.{io_type}.training.bin"

    # Check if the kmeans file exists, otherwise use training
    if os.path.exists(kmeans_file):
        trainingData, num_vectors, dims = load_embeddings(kmeans_file)
    elif os.path.exists(training_file):
        trainingData, num_vectors, dims = load_embeddings(training_file)
    else:
        raise FileNotFoundError(f"Neither {kmeans_file} nor {training_file} found!")

    # Set number of neighbors (k) based on dataset type
    if os.path.exists(kmeans_file):
        # Use single centroid for kmeans data
        k = 1
    else:
        k = min(20, max(10, int(math.log2(num_vectors)) + 5))

    # Initialize HNSW index
    # 'l2' = Euclidean distance
    index = hnswlib.Index(space="l2", dim=dims)

    # TODO Research ef_construction
    # Build the index
    # ef_construction : Controls build speed/accuracy trade-off
    # M:  Number of bidirectional links per node
    index.init_index(max_elements=len(trainingData), ef_construction=200, M=16)

    # Add data to the index
    index.add_items(trainingData)

    # Destructering labels and distances from the nearest neighbors query
    labels, distances = index.knn_query(query, k=k)

    # Ends timing for the entire function
    endTotal = time.perf_counter()

    centroids = trainingData[labels[0]]

    return {
        "centroids": centroids.tolist(),
        "distances": distances[0].tolist(),
    }

# TODO Remove hardcoded testing stuff
# # This is for testing purposes only, delete
# io_type = 'input'
# model_type = 'lexical'
# query = np.random.rand(1, 384).astype(np.float32)
# HNSW(io_type, model_type, query)

# Checks that the file is run directly, not as an import
if __name__ == "__main__":
    # Error handling to check that there are 3 arguments and 1 script
    if len(sys.argv) != 4:
        # Print the error
        print(
            json.dumps(
                {
                    "error": "Usage: python3 pythonHNSW.py <io_type> <model_type> <query_json>"
                }
            )
        )
        sys.exit(1)
    try:
        # assign the value of result to the evaluated result of invoking HNSW with the 3 input arguments
        result = HNSW(sys.argv[1], sys.argv[2], sys.argv[3])
        # Returns the value of result to javascript file
        print(json.dumps(result))
        # Catch all error handling
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
