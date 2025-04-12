from datasets import Dataset, concatenate_datasets
import os
import json
import numpy as np
from datetime import datetime
from util.pythonTrainingEmb import resolve_io_column

# * Writes shared scalar metrics (like character length, entropy, etc.) for training data
# * One file is created per metric per I/O type (e.g., input.characterLength.training.scalar.jsonl)
def write_shared_scalar_metrics(data_path, io_type, io_type_name):
    # Load all `.arrow` files from the provided dataset directory
    arrow_files = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith(".arrow")
    ]

    # Combine them into a single dataset
    dataset_parts = [Dataset.from_file(fp) for fp in sorted(arrow_files)]
    dataset = concatenate_datasets(dataset_parts)

    # Use our helper to extract the proper column of input/output text
    batch_texts = resolve_io_column(dataset, io_type_name)

    # Loop through every text item and compute shared scalar values
    for text in batch_texts:
        timestamp = datetime.now().isoformat()  # ISO timestamp for tracking

        # --- Compute Shared Metrics ---

        # Number of characters in the input/output
        character_length = len(text)

        # Character entropy (measures repetition vs. diversity)
        counts = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        character_entropy = -sum(
            (count / len(text)) * np.log2(count / len(text))
            for count in counts.values()
        )

        # Average word length
        words = text.split()
        avg_word_length = (
            sum(len(word) for word in words) / len(words) if words else 0
        )

        # Ratio of punctuation to total characters
        punctuation_density = sum(1 for c in text if c in '.,!?;:') / len(text) if len(text) > 0 else 0

        # Ratio of uppercase letters to total characters
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0

        # Bundle all metrics together
        shared_metrics = {
            "characterLength": character_length,
            "characterEntropy": character_entropy,
            "avgWordLength": avg_word_length,
            "punctuationDensity": punctuation_density,
            "uppercaseRatio": uppercase_ratio,
        }

        # --- Write each metric to its own scalar file ---
        for metric, value in shared_metrics.items():
            # File format: input.avgWordLength.training.scalar.jsonl
            file_path = f"data/scalars/{io_type}.{metric}.training.scalar.jsonl"

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Append one JSONL line per metric, per text
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": timestamp,
                    "metrics": {metric: value}
                }) + "\n")
