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
# Allow error logging for testing purposes
import traceback

def HNSW(io_type, model_type, query, baseline_type):

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
                print(f"⚠️ Header says: {num_vectors} vectors of {dims} dims", file=sys.stderr)
                print(f"⚠️ But only {len(data)} floats were read", file=sys.stderr)
                raise ValueError("Data is incorrect - size mismatch")

            reshaped_data = data.reshape(num_vectors, dims)
            # Check to make sure this is not the first write AND remove the new entry for the rolling dataset
            if baseline_type == 'rolling' and num_vectors != 1:
            # Keep only the most recent 1,000 entries when we are reading from rolling.
                reshaped_data = reshaped_data[-1000:-1]

            # If we're in training, we'll only read the most recent 10,000. This will not cause problems with existing training data sets greater than that amount because anything 10,000 and above will receive k-means clustering. In this case we'll be reading from the rolling file, which can get infinitely large, theoretically.
            elif baseline_type == 'training':
                reshaped_data = reshaped_data[:10000]
            
            # Set num vectors equal to the actual number we pulled from the reshaped data
            num_vectors = len(reshaped_data)
                
            # Returns the vectors from the binary file
            return reshaped_data, num_vectors, dims

    # TODO: These paths are relative from their execution directory, so this may not work in production
    # Define file paths
    kmeans_file = f"data/kmeans/{model_type}.{io_type}.{baseline_type}.kmeans.bin"
    data_file = f"data/vectors/{model_type}.{io_type}.{baseline_type}.bin"
    # after setting our filepaths, we check if the path doesn't exist and the baseline type is training. since kmeans gets used first, this will only matter when there isn't training data at all.
    if (baseline_type == 'training' and not os.path.exists(data_file)):
        data_file = f"data/vectors/{model_type}.{io_type}.rolling.bin"
    
    # Check if the kmeans file exists, otherwise use training
    if os.path.exists(kmeans_file):
        data, num_vectors, dims = load_embeddings(kmeans_file)
    elif os.path.exists(data_file):
        data, num_vectors, dims = load_embeddings(data_file)
        if (num_vectors < 10):
            return {
                "centroids": data.tolist(),
                "distances": None,
            }
    else:
        raise FileNotFoundError(f"Neither {kmeans_file} nor {data} found!")

    # Set number of neighbors (k) based on dataset type
    if os.path.exists(kmeans_file):
        # Use single centroid for kmeans data
        k = 1
    else:
        k = min(20, max(10, int(math.log2(num_vectors)) + 5))

    # TODO: Maybe there is a better way to scale these values other than linearly
    # Set ef_construction to len(data)-1 when len(data) is less than 200
    if (len(data) < 200):
        ef_construction = max(1, len(data) - 1)
    else:
        ef_construction = 200
        
    # Set M to len(data)-1 when len(data) is less than 16
    if (len(data) < 16):
        M = max(1, len(data) - 1)
    else:
        M = 16
    # Initialize HNSW index

    # 'l2' = Euclidean distance
    index = hnswlib.Index(space="l2", dim=dims)

    # Build the index
    # ef_construction : Controls build speed/accuracy trade-off
    # M:  Number of bidirectional links per node
    index.init_index(max_elements=num_vectors, ef_construction=ef_construction, M=M)

    # Add data to the index
    index.add_items(data)

    # Destructuring labels and distances from the nearest neighbors query
    labels, distances = index.knn_query(query, k=k)

    # Ends timing for the entire function
    endTotal = time.perf_counter()

    centroids = data[labels[0]]

    return {
        "centroids": centroids.tolist(),
        "distances": distances[0].tolist(),
    }

# Checks that the file is run directly, not as an import
if __name__ == "__main__":
    # Error handling to check that there are 3 arguments and 1 script
    if len(sys.argv) != 5:
        # Print the error
        print(
            json.dumps(
                {
                    "error": "Usage: python3 pythonHNSW.py <io_type> <model_type> <query_json> <baseline_type>"
                }
            )
        )
        sys.exit(1)
    try:
        # assign the value of result to the evaluated result of invoking HNSW with the 3 input arguments
        result = HNSW(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        # Returns the value of result to javascript file
        print(json.dumps(result))
        # Catch all error handling
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
