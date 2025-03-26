import os
import json
from datasets import load_dataset

# Load dataset
dataset_name = "SmallDoge/SmallThoughts"
dataset = load_dataset(dataset_name)

# Create output directory if it doesn't exist
output_dir = "smallthoughts_data"
os.makedirs(output_dir, exist_ok=True)

# Define output file path
output_file = os.path.join(output_dir, "dataset.json")

# Extract data into a list of dictionaries
data_list = []
for i, entry in enumerate(dataset["train"]):
    data_list.append({
        "input": entry["problem"].strip(),
        "output": entry["solution"].strip()
    })

    if i % 1000 == 0:  # Print progress every 1000 entries
        print(f"Processed {i+1} entries...")

# Save all data as a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"Dataset saved in JSON format: {output_file}")
