# Import helper function to create kmeans of data
import pythonKMeans

# This is good for vectors/matrices
import numpy as np

from transformers import AutoModel, AutoTokenizer

import torch

from datasets import Dataset

# Allows the use of time functions
import time

import os


def trainingEmb(model_type, model_name, data_path, io_type, io_type_name):

    # Starts the total function timer
    startTotal = time.perf_counter()

    # Set device (MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load model components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Check if dataset exist
    if data_path:
        print(f"Loading dataset from local cache: {data_path}")
        dataset = Dataset.from_file(data_path)
    else:
        raise ValueError(print("Could Not Retrieve Dataset"))

    # This prevents the creation of gradients
    @torch.no_grad()
    # Embedding function
    def embed_data(data):
        # Tokenizes the input data
        inputData = tokenizer(
            # TODO Look up how truncation and max_length works
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            outputData = model(**inputData)
        return outputData.last_hidden_state.mean(dim=1).cpu().numpy()

    # Embed Data
    print(f"\nEmbedding {io_type}s...")
    # Initialize an empty list to store all input embeddings
    embeddings = []
    # Set the number of examples to process at once (smaller = less memory, larger = faster)
    batch_size = 10

    # Loop through the dataset in batches
    # range(start, stop, step) creates a sequence from 0 to len(dataset) in batch_size increments
    for i in range(0, len(dataset), batch_size):
        # Get a batch of input texts:
        # dataset[i : i + batch_size] slices the dataset to get current batch
        # [io_type_name] selects just the input column
        batch = dataset[i : i + batch_size][io_type_name]
        # Convert the batch of texts to embeddings using our embedding function
        emb = embed_data(batch)
        # Add this batch's embeddings
        embeddings.append(emb)
        # Print progress every 10 batches
        if i % (batch_size) == 0:
            print(f"Processed {min(i+batch_size, len(dataset))}/{len(dataset)}")

    embeddings = np.concatenate(embeddings)

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    if (len(embeddings) < 10, 000):
        # Assign the number of vectors for the training data
        num_vectors = embeddings.shape[0]
        # Assign the dimensions of each vector
        dims = embeddings.shape[1]

        # Create header bytes (8 bytes total)
        header_bytes = np.array([num_vectors, dims], dtype=np.uint32).tobytes()

        # Write to file (header first, then data)
        with open(f"data/{model_type}.{io_type}.training.bin", "wb") as f:
            # Write 8-byte header first
            f.write(header_bytes)
            # Then write the data
            embeddings.astype(np.float32).tofile(f)
    else:
        kMeansEmbedding = pythonKMeans.kMeansClustering(embeddings)
        # Save the embeddings
        embeddings.astype(np.float32).tofile(
            f"data/{model_type}.{io_type}.kmeanstraining.bin"
        )

    # TODO Remove this before going live
    # # This is for testing purpose only, delete
    # kMeansEmbedding = pythonKMeans.kMeansClustering(embeddings)
    # #  Save the embeddings for testing only, delete
    # kMeansEmbedding.astype(np.float32).tofile(f"data/{model_type}.{io_type}.kmeanstraining.bin")

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Elapsed: {endTotal - startTotal:.6f} seconds")

    return
