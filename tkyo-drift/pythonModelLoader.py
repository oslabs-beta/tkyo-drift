# This is good for vectors/matrices
import numpy as np

from transformers import AutoModel, AutoTokenizer

import torch

from datasets import Dataset

# Allows the use of time functions
import time

MODEL = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
DATASET_PATH = "./data/small_thoughts-train.arrow"


def modelLoader(model_name, dataSet_Path, input_name="input", output_name="output"):
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
    if dataSet_Path:
        print(f"Loading dataset from local cache: {dataSet_Path}")
        dataset = Dataset.from_file(dataSet_Path)
    else:
        raise ValueError(print("Could Not Retrieve Dataset"))

    print(len(dataset))
    print(dataset[0][input_name])

    @torch.no_grad()
    # Embedding function
    def embed_data(data):
        # Tokenizes the input data
        inputData = tokenizer(
            data, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputData = model(**inputData)
        return outputData.last_hidden_state.mean(dim=1).cpu().numpy()

    # Embed Inputs
    print("\nEmbedding inputs...")
    # Initialize an empty list to store all input embeddings
    input_embeddings = []
    # Set the number of examples to process at once (smaller = less memory, larger = faster)
    batch_size = 8

    # Loop through the dataset in batches
    # range(start, stop, step) creates a sequence from 0 to len(dataset) in batch_size increments
    for i in range(0, 100, batch_size):
        # Get a batch of input texts:
        # dataset[i : i + batch_size] slices the dataset to get current batch
        # [input_name] selects just the input column (e.g., "problem" texts)
        batch = dataset[i : i + batch_size][input_name]
        # Convert the batch of texts to embeddings using our embedding function
        embeddings = embed_data(batch)
        # Add this batch's embeddings
        input_embeddings.append(embeddings)
        # Print progress every 10 batches
        if i % (batch_size) == 0:
            print(f"Processed {min(i+batch_size, len(dataset))}/{len(dataset)}")

    input_embeddings = np.concatenate(input_embeddings)

    # Embed Outputs
    print("\nEmbedding outputs...")
    # Initialize an empty list to store all input embeddings
    output_embeddings = []
    # Set the number of examples to process at once (smaller = less memory, larger = faster)
    batch_size = 8

    # Loop through the dataset in batches
    # range(start, stop, step) creates a sequence from 0 to len(dataset) in batch_size increments
    for i in range(0, 100, batch_size):
        # Get a batch of input texts:
        # dataset[i : i + batch_size] slices the dataset to get current batch
        # [input_name] selects just the input column (e.g., "problem" texts)
        batch = dataset[i : i + batch_size][output_name]
        # Convert the batch of texts to embeddings using our embedding function
        embeddings = embed_data(batch)
        # Add this batch's embeddings
        output_embeddings.append(embeddings)
        # Print progress every 10 batches
        if i % (batch_size) == 0:
            print(f"Processed {min(i+batch_size, len(dataset))}/{len(dataset)}")

    output_embeddings = np.concatenate(output_embeddings)

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Elapsed: {endTotal - startTotal:.6f} seconds")

    print(input_embeddings[0])
    return {
        "input_embeddings": input_embeddings,
        "output_embeddings": output_embeddings,
        "model": model,
        "tokenizer": tokenizer,
    }

# Invoke the modelLoader function
modelLoader(MODEL, DATASET_PATH, "problem", "solution")
