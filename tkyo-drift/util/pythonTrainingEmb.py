# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
# Import helper function to create kmeans of data
from util import pythonKMeans
# This is good for vectors/matrices
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from datasets import Dataset, concatenate_datasets
# Allows the use of time functions
import time
import json
from datetime import datetime

import os
# TODO: This has a chance of failing during write, but will fail silently.
# ! We should implement a solution, like writing to a temp file, and then renaming the temp file after completion
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
        print(f"Loading all .arrow files from: {data_path}")
        
        arrow_files = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".arrow")
        ]

        if not arrow_files:
            raise ValueError("No .arrow files found in the provided directory.")

        dataset_parts = [Dataset.from_file(fp) for fp in sorted(arrow_files)]
        dataset = concatenate_datasets(dataset_parts)
    else:
        raise ValueError(print("Could Not Retrieve Dataset"))

    # This prevents the creation of gradients
    @torch.no_grad()
    # TODO: This section effectively means we are only embedding the first 512 tokens, and all data thereafter is lost
    # ! Should be ok for most AI workflows, but this will be a problem for ones that take large text inputs
    # truncation	Cuts off long sequences (from the end)
    # max_length	Upper bound for number of tokens
    # padding	Adds [PAD] tokens to match batch size
    # Embedding function
    def embed_data(data):
        # Tokenizes the input data
        inputData = tokenizer(
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
        batch_raw = dataset[i : i + batch_size]

        # Use helper to resolve column names 
        batch = resolve_io_column(batch_raw, io_type_name)


        # Convert the batch of texts to embeddings using our embedding function
        emb = embed_data(batch)

        # Compute and log scalar metrics for each item in batch
        scalar_lines = []
        for j, vector in enumerate(emb):
            text = batch[j]

            # Norm of the vector
            norm = float(np.linalg.norm(vector))

            # Raw text length
            text_length = len(text)

            # Token length (use tokenizer to match JS side)
            token_length = len(tokenizer.encode(text))

            # Character-level entropy
            counts = {}
            for c in text:
                counts[c] = counts.get(c, 0) + 1
            entropy = -sum((count / len(text)) * np.log2(count / len(text)) for count in counts.values())

            # Average word length
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

            # Punctuation density
            punctuation_density = sum(1 for c in text if c in '.,!?;:') / len(text) if len(text) > 0 else 0

            # Uppercase ratio
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0

            # Store as JSONL
            scalar_lines.append(json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics": {
                    "norm": norm,
                    "textLength": text_length,
                    "tokenLength": token_length,
                    "entropy": entropy,
                    "avgWordLength": avg_word_length,
                    "punctuationDensity": punctuation_density,
                    "uppercaseRatio": uppercase_ratio
                }
            }) + "\n")

        # Write to .jsonl file (append mode)
        os.makedirs("data/scalars", exist_ok=True)
        with open(f"data/scalars/{model_type}.{io_type}.training.scalar.jsonl", "a", encoding="utf-8") as f:
            f.writelines(scalar_lines)


        # Add this batch's embeddings
        embeddings.append(emb)

        # Print progress every 10 batches
        if i % (batch_size) == 0:
            print(f"Processed {min(i+batch_size, len(dataset))}/{len(dataset)}")

    embeddings = np.concatenate(embeddings)

    # Create data directories if they doesn't exist
    os.makedirs("data/vectors", exist_ok=True)
    os.makedirs("data/kmeans", exist_ok=True)

    if (len(embeddings) < 10000):

        # Assign the number of vectors for the training data
        num_vectors = embeddings.shape[0]

        # Assign the dimensions of each vector
        dims = embeddings.shape[1]

        # Create header bytes (8 bytes total)
        header_bytes = np.array([num_vectors, dims], dtype=np.uint32).tobytes()
    

        # Write to file (header first, then data)
        with open(f"data/vectors/{model_type}.{io_type}.training.bin", "wb") as f:

            # Write 8-byte header first
            f.write(header_bytes)

            # Then write the data
            embeddings.astype(np.float32).tofile(f)
    else:
        kMeansEmbedding = pythonKMeans.kMeansClustering(embeddings)

        # Assign the number of vectors for the training data
        num_vectors = kMeansEmbedding.shape[0]

        # Assign the dimensions of each vector
        dims = kMeansEmbedding.shape[1]

        # Create header bytes (8 bytes total)
        header_bytes = np.array([num_vectors, dims], dtype=np.uint32).tobytes()
        


        # Write to file (header first, then data)
        with open(f"data/kmeans/{model_type}.{io_type}.training.kmeans.bin", "wb") as f:

            # Write 8-byte header first
            f.write(header_bytes)

            # Then write the data
            kMeansEmbedding.astype(np.float32).tofile(f)

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Elapsed: {endTotal - startTotal:.6f} seconds")

    return

def resolve_io_column(batch, io_type_name):
    try:
        if hasattr(batch, "column_names") and io_type_name in batch.column_names:
            return batch[io_type_name]

        # Handle batch = dict of lists (Hugging Face batch)
        sample = list(batch.values())[0][0]

        return [row[io_type_name] for row in [
    {k: v[i] for k, v in batch.items()} for i in range(len(next(iter(batch.values()))))
]]

    except Exception as e:
        raise ValueError(f"Could not extract '{io_type_name}' from dataset: {e}")
