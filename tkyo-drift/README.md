TODO: Add a section to readme about embedding max length, tokenization, and silent truncation

# AI Temporal Knowledge Yield Output Drift Tracker (TKYO Drift) 

TKYO Drift is a lightweight, transparent drift tracking library for AI workflows. It embeds I/O Pairs and compares them to a configurable baseline to detect drift in semantic, conceptual, or lexical meaning over time.

At the time of writing, this tool is only able to ingest data from mono modal text generation AI workflows.  

## Overview

This tool is designed for performance-critical environments and was designed to be operable in production environments. The tkyoDrift() function can be inserted into a text generation workflow as an asynchronous function to minimize performance impacts in production settings. It operates quasi-locally with Python dependencies using the a JS transformers library to support both real-time logging of new I/O data, and batch ingestion of I/O training data. Drift is tracked via cosine similarity against baselines, and stored in compact binary files.

## Drift Analysis Flow

For each individual input and output, the following workflow is executed in roughly ~500ms. 

1. Generate embeddings for both input and output.
2. Save them to `.bin` files (except for training mode).
3. Load historical vectors from disk.
4. Compute baseline vectors as an average of past embeddings.
5. Calculate cosine similarity.
6. Append results to `drift_log.csv`.

```
Note that the size of input/output text, embedding dimensions, and how many embedding models are chosen will influence the speed of the workflow. Regardless, if tkyoDrift() is called asynchronously, it should not impact your workflow unless you expect a sustained high volume of user inputs per second.
```
# How do you install this thing?

TODO: Add an install section

# How do you use this thing?

You can interact with this library in 3 ways; 
- Dispatch a one-off I/O pair to `tkyoDrift()`
- Dispatch sequential training data through a batch upload to `tkyoDriftSetTraining()`*
- Request a CLI print out of the log's summary using `npx tkyoDrift -#` (where # is a number)

```
* Do this from a strong PC, and then copy your data into the appropriate folders. Due to a number of factors (I/O Length, CUDA access, memory, cpu cores, training data size) this process can take an exceptionally long time to complete.
```
## One-off Ingestion

`tkyoDrift.js(input, output, depth)` handles individual I/O pairs for drift comparison. It:

- Accepts two text strings as `input` and `output` parameters.
- Accepts an optional 3rd command for the depth* of the I/O pair to enable tracking of nested chain of thought I/O pairs.
- Embeds serially using as many models as you specify.
- Appends to the rolling data file (only).
- Reads from the file to load a baseline for comparison.
- Computed the drift score using cosine similarity.

```
* Depth tracking is useful when you have multiple chain of thought workflows, and want to track drift per depth level. Just pass a depth argument for how nested each I/O is. For example, your model might have Human -> AI -> AI, where the first input and last output would be depth = 0, and the middleware AI's input and output would be depth = 1.
```

## Training Ingestion

`tkyoDriftSetTraining.js(filepath)` handles full dataset ingestion for baseline creation. It:

- Accepts an array of `{ input, output }` objects.
- Embeds in chunks of 100 for performance.
- Clears old training files before writing new ones.
- Locks files after saving.
- Optionally runs garbage collection between chunks.

```
IMPORTANT! The set training data file should ONLY be run when you intend on replacing the existing tkyoDrift training data file set. This command will obliterate any existing tkyoDrift training files. 

<<< There should only be ONE set of ideal embeddings for your training data. >>>
``` 

## Logging

Results are stored in two CSV files (`COS_log.csv` & `EUC_log.csv`) with dynamic headers. Each one-off run appends two rows to each file: one for input and one for output. Keep in mind that training data is not added to the log, as the assumption is that your training baseline is what we compare against to measure drift.

### Format

For the cosine similarity log:
```
ID, DEPTH, TIMESTAMP, I/O TYPE, SEMANTIC ROLLING COS, SEMANTIC TRAINING COS, CONCEPT ROLLING COS...
```
For the euclidean distance log:
```
ID, DEPTH, TIMESTAMP, I/O TYPE, SEMANTIC ROLLING EUC, SEMANTIC TRAINING EUC, CONCEPT ROLLING EUC...
```
- Cosine similarities and euclidean distances are recorded per model and baseline type.
- Additional metadata like `depth` and UUIDs are included for tracking.
- UUIDs are shared for the input and output, so that you can review your input/output data logs to inspect which I/O pairs are causing drift.
- Neither the log, nor the binary files, contain your users input or ai outputs. This data is not necessary to calculate drift, and it's exclusion is an intentional choice for data privacy.

```
Note: if you add or remove model types to the tkyoDrift tracker, the log will break. Please ensure you clear any existing logs after altering the embedding model names. What we mean here, is that if you change your lexical model from "lexical" to "linguistic". When writing to the log, the makeLogEntry method of the Drift Class would work, but the log Parser would fail. 

Keep in mind, however, you can change models any time you like, though that will brick drift calculations for a different reason; Your inputs/outputs we be embedded with dissimilar methods, which would lead to inaccurate drift calculations.
```

## CLI Tools

TODO: Add the command we need people to enter to trigger this log here after the NPM package is built.

### `printLogCLI.js`

Parses `COS_log.csv` and displays violation counts and average cosine similarities over a selected number of days. Uses a color-coded table (green/yellow/red) to show severity of drift. Thresholds are set in this file, and should be adjust to your expected precision needs.

```
Note: The first record you enter into this system will always show that there is 0 drift when compared against the rolling data set. This is because the rolling dataset will be compared against itself at that point and there will be no drift to detect. This is a known issue, and was an intentional choice as the alternative would be to exclude a write to the `COS_log.csv` and `EUC_log.csv` logs on first write. 

If this bothers you, you can remove line 2 from the COS and EUC logs after you use this system at least twice.
```

### `printScalarCLI.js`

Parses the scalar jsonl files to calculate scalar distributions across the training and rolling datasets and delta mean and delta standard deviation between the two distributions. Uses a color-coded table (green/yellow/red) to show severity of drift. Thresholds are set in this file, and should be adjust to your expected precision needs.

The scalar metrics the system is currently tracking is listed below:

| Metric               | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `norm`               | Vector magnitude (captures changes in embedding length/energy)|
| `textLength`         | Raw character count of the input/output text                  |
| `tokenLength`        | Number of tokens (based on model tokenizer)                   |
| `entropy`            | Character-level entropy (measures information density)        |
| `avgWordLength`      | Average word length (indicates language complexity)           |
| `punctuationDensity`| Ratio of punctuation to characters (captures tone/stylistics) |
| `uppercaseRatio`     | Ratio of uppercase letters (detects emphasis or acronyms)     |

```
Note: Without batch embedding your training data, you will be unable to view scalar metrics. However, we will still collect scalar metrics for your rolling data in case you add training data later.
```

# Architecture

### Embedding Models

TKYO Drift uses remote embedding models on HuggingFace.co for inference using the Xenova Transformers library in javascript or the python native hugging face library in python. By default, the system operates in JavaScript using a lightweight transformer pipeline, with Python scripts injected as required to improve speed when performing batched operations. The python equivalent embedding pipeline allows for CPU usage by default if your system has the appropriate CUDA drivers.

- `all-MiniLM-L12-v2`: Used for semantic drift or changes in tone or communication style.
- `e5-base`: Used for concept drift or changes in topic or intent.
- `all-MiniLM-L6-v2`: Used for lexical drift or changes in word choice.

Note that while L6 is a subset of L12, it is also the case that lexical drift is a subset of semantic drift. This model can be disabled if you believe that MiniLM-L12 is comprehensive enough to provide drift tracking for both types. This speeds up one-off and batched operations by about 10%.

### Xenova/Hugging Face Transformer Library

While the Xenova transformer library is a javascript equivalent of the Hugging Face python transformer library, the primary difference is that the former was made for JS while the latter was made for Python. The HF library allows for GPU acceleration, which is why it was chosen for batched calls. In either case, we are using the same transformer library for the same purpose. As such, Xenova/Hugging Face Transformers were chosen as the preferred choice of transformer libraries because:

- Embedding models are quite large, and including a wget or some other form of downloading models to run locally would require dealing with user authentication for the huggingface.com site, which we wanted to avoid. 
- People have good internet now, mostly, so we can get away with streaming the model conclusions to the workflow environment without dealing with a local model.
- Using wasm for the embedding pipeline lets people hot swap models without having to replace safetensors.

### Drift Types

Drift is detected across combinations of:

- `modelType`: semantic, concept, lexical
- `ioType`: input, output
- `baselineType`: rolling, training

This results in either six or twelve cosine similarity and euclidean distance comparisons per I/O pair, depending on whether or not training data was provided.
- All embeddings are pooled across all input tokens to get mean values from the input and normalized for all models except for the conceptual drift type.
- Models are loaded sequentially and cached globally to reduce loading time.

Note that once the execution context window closes for the processing I/O pair, models are naturally unloaded. Unless you feel inclined to download and store the models locally in your production pipeline, this is a necessary and unavoidable ~200ms workflow speed penalty.

```
Drift detection in the scalar metrics is ONLY available when a training dataset is provided, as scalar metrics are comparisons in distribution shape. In other words, without a training distribution to compare against the rolling distribution, there is no comparison to make.
```

### Baseline Types

- `Rolling`: A sliding baseline using the most recent N (default 1,000) examples.

The rolling baseline represents the accumulation of inputs and outputs as they are triggered by the production pipeline. As I/O pairs are dispatched to the drift analysis workflow, each I/O pair is saved to the rolling baseline file. When drift calculations are generated, they exclude the newest I/O pair for that individual calculation but will include them in all subsequent operations. The total number of I/O pairs to be included in drift analysis is a configurable variable located in the pythonHNSW.py file. 

- `Training`: A fixed baseline built once from a full dataset.

The training baseline represents the set of inputs used to generate the initial model AI responses, which are ingested by running the 'tkyoDriftSetTraining.py' script. This script performs a batch analysis of each I/O pair used in the training set and creates an artificially 'locked' training file that can not/will not be be appended to over time. Theoretically, you should only need to update this when you have retrained you model.

- `Hybrid`: Use the rolling file to provide first K (default 10,000) and last N (default 1,000) simultaneously.

In the event that there is no training data supplied to the system, the Drift analyzer will use the first N values entered into the rolling file to represent mock 'training' data, while the last K (default of 10,000) values represent the 'rolling' data. This allows us to compare drift against an anchor point to see drift over time, while still having a rolling window to see shock impacts to the system caused by new concepts/semantics/lexicon.


### Binary Embedding Storage

Embeddings are saved in `.bin` files using float16 for efficient storage. This minimizes disk I/O and enables fast appending. Note that math calculates are still performed in float32 after float16 conversions after disk reads. While this adds noise and make drift detection less precise, it should only impact drift detection after 6 decimal points.

- Each file is named:  
  `{modelType}.{ioType}.{baselineType}.bin`
  
This yields `(models * I/Os * baselines)` file combinations, and at the minimum should represent two files if you are using a single drift model for an I/O pair with only the rolling baseline. By default this library is configured to run with 3 drift types for an I/O pair, and will have 6 rolling files. This would become 12 should you upload training data. 

At the time of writing, the default models in this library have either 768 or 384 dimensions per input.  

| Model                          | Purpose                         | Dimensions | Bytes per Input (float16) | File Size (10,000 inputs) |
|-------------------------------|----------------------------------|------------|----------------------------|----------------------------|
| `Xenova/all-MiniLM-L12-v2`    | Semantic (communication method) | 384        | 768 bytes                 | ~7.5 MB                   |
| `Xenova/e5-base`              | Concept (communication intent)  | 768        | 1,536 bytes               | ~15.0 MB                  |
| `Xenova/all-MiniLM-L6-v2`     | Lexical (syntax)                | 384        | 768 bytes                 | ~7.5 MB                   |

Note: 1 MB = 1,048,576 bytes (binary MB), but here we're rounding to 1 MB = 1,000,000 bytes for simplicity.

Scalar files are negligibly large, and even with 1 million records, they should take less than 250 MBs and the Log files themselves are miniscule. 

```
The rolling files have no upper limit on their size, and will require manual pruning eventually depending on your workflow's throughput. Incidentally, if you do not have access to your training data (you may be using a 3rd party model without a published data set) you may benefit from renaming your rolling files to training files after you have accumulated 10,000 entries.
```

### IO Write/Read Methods

File writing/reading is performed using `fs.promises.writeStream` and `fs.promises.readStream` due to the obscene number of floats we need to read, parse and then calculate against, which necessitates that vectors be constructed during the read process to avoid memory overflows. This is an intentional tradeoff of speed for deployability. If you know your system has the memory to spare, IO ops can be improved by replacing write/read streams with loads-to-memory.

## HNSW Indexing

HNSW (Hierarchical Navigable Small World) allows us to support approximate nearest neighbor queries on stored embeddings. This allows for:

- Fast lookup of drift clusters
- Identification of nearest historic I/O pairs
- Calculation of sub cluster centroids within general text AI model training data.
- N(logN) lookup across (theoretically) millions of embeddings.
- Tunable accuracy and recall settings.

# The Math

The core calculation behind drift tracking is **cosine similarity**, which evaluates the angle between two vectors in high-dimensional space, ignoring their magnitude. This is ideal for comparing semantic or conceptual distance between two pieces of text.

## Cosine Similarity & Euclidean Distance

Cosine similarity between two vectors **A** and **B** is calculated as:

Where:
- `A · B` is the dot product of vectors A and B.
- `||A||` is the magnitude (L2 norm) of vector A.
- `||B||` is the magnitude of vector B.

The result is a value between -1 and 1. For normalized embedding vectors (as used here), the output is always between 0 and 1:
- `1.0` → Identical direction (no drift)
- `0.0` → Orthogonal (maximum drift)

Normalization ensures magnitude doesn’t influence the result, so only the *direction* of the vector matters. Additionally, we are calculating the Euclidean Distance. This metric is not scale-invariant and is typically larger in magnitude. It’s useful in conjunction with cosine similarity to detect both directional and magnitude-based drift.

## What B Represents

In the context of TKYO Drift, **B** is the *baseline vector* against which a new embedding (**A**) is compared. It represents the average of historical embeddings from either:

- The **rolling baseline**, which reflects the most recent N inputs/outputs in production.
- The **training baseline**, which reflects embeddings generated from your original training dataset.
- Or both (in hybrid mode), where the system simulates a static baseline from the start of the rolling file.

This averaged baseline acts as a reference point. If the direction of the new embedding (A) starts to diverge from B, it indicates a potential drift in communication, concept, or syntax.

## Centroid Calculation

Since the B value in question is the basis for the drift comparison, it is imperative that the B value here be the average centroid node of whatever sub cluster of nodes exists within a training or rolling data set closest to the A value.

This is where K-Means analysis comes into play, allowing us to condense a large input dataset into a smaller set by identifying how many sub clusters exist, and calculating each of their respective centroids.

From this smaller set of centroids, we use HNSW to to find the closest centroid for the B value, which is then used to calculate the cosine similarity and euclidean distance.

Notably, this is a tradeoff between accuracy and speed, as KMeans cluster analysis will generate a centroid for each cluster and not provide the actual nearest neighbor. If this is a problem for your workflow, you can disable the KMeans analysis to always find the nearest neighbor from the training set.

This system uses `(num_of_clusters = int(np.sqrt(num_vectors / 2)))` to determine the number of clusters to generate, as we do not have the ability to use the elbow method to determine the proper value for K.

# Future Iterations

For this project there are a number of ways this platform could be made better, including:

### Rolling Data Selection

The rolling file currently uses a fixed number of records when loaded during the readFromBin method on the DriftModel class, this fixed number may not be a flexible enough method of selecting the number of records to compare for rolling drift. It would be an improvement to modify this so that it uses entries from the last N days instead. 

Building this would require adding a data to the binary file write method to store when a file was added, or replacing the binary file writer entirely and switching to a different file storage format. Regardless, at the time of this readme writeup, there is no date associated with stored vectors and so there is no way to have the rolling file use the last N days of records instead of a fixed count.

### Rolling Data Selection

The current implementation sets K using a heuristic: K = int(sqrt(num_vectors / 2)), which balances clustering granularity with speed. While the elbow method offers a more statistically grounded way to choose K by evaluating clustering performance across several K values, it is computationally intensive.

Implementing the elbow method would require running KMeans multiple times and analyzing metrics like SSE or silhouette scores. Given our real-time and batch constraints, we avoid this due to diminishing accuracy gains (often logarithmic) versus increased computational cost (often linear to exponential with larger datasets).

### Depth Tracking

While being able to track multiple chains of thought in your AI workflow is an important part of measuring drift, the current implementation of depth tracking suffers from a critical drawback in that COS similarity, EUC distance, and scalar metrics are NOT calculated for each depth band. 

Resolving this would require a major refactor of the calculation of drift metrics so that they only compare against other entries of the same depth. Alternatively, it would require writing to individual logs for each depth. In any case, the current functionality may unintentionally pollute the drift scores as I/O from different depths are included in the calculations for COS similarity/EUC distance/Scalar metrics.

### Python vs Javascript

This project was initially built as a pure javascript project to enable wider deployment, but various functions and libraries were originally built, and intended to be used in python. As a result, this project was refactored after an initial test build to include a javascript pipeline for individual embeddings and a python version for batched embeddings.

What this means is that the tkyoDriftSetTraining.py file and the tkyoDrift.js processes are functionally duplicates of each other with the exception that the former is explicitly meant to be called once for a batch, while the later is meant to be invoked on every new input.

This is fine as it is, but since many javascript libraries are just python scripts wearing a disguise, it would be ideal to rebuild this entire platform in python with a javascript NPM package to install it, and a javascript function hook to pass data into it. This would allow this system to avoid unnecessary conversion from javascript into python to execute AI embeddings, calculate K means, or generate the HNSW index.