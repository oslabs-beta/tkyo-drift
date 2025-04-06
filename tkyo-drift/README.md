TODO: Add an explanation for why the first value will never show drift

TODO: Add an install section

TODO: Add section to readme about how to determine the amount of clusters in pythonKMeans.py (num_of_clusters = int(np.sqrt(num_vectors / 2)))

TODO: Add a section to readme about embedding max length, tokenization, and silent truncation

TODO: Changing embedding model names (for example, from 'lexical' to 'bagOfWords') will brick the log. Warn the people.

TODO: Explain limitations of the depth counter and why its only a partial solution.

TODO: We are currently using a Xenova and Hugging Face transformers library split, with xenova in JS (one offs) and HF in PY (batched calls w/ CUDA). We need to explain why we are doing this. As a side note, Xenova IS huggingFace, but not maintained by hugging face, its just the JS version of HF's python library maintained by Xenova

TODO: We need to mention that the rolling file will expand infinity, and should be deleted occasionally.

TODO: We need to remove references to First N vs Last N in training/rolling. Wait a minute... if there is no training data, do we still want to use the rolling files? If yes, do we need to set an upper limit on the first 10k in rolling to represent the training data? That functionality is not in this code at the moment.

TODO: We should probably mention that pythonTrainingEmb.py is a functional duplicate/stripped down version of tkyodrift because it does all the same things but in batches?

TODO: Add a warning about training batch embedding times, and how slow it is without a graphics card.

TODO: We should add some advice that running tkyoDriftSetTraining.py should be done in a dev environment, and then the data should be moved into the production data folder.

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




# How do you use this thing?

You can interact with this library in 3 ways; 
- Dispatch a one-off I/O pair to `tkyoDrift()`
- Dispatch sequential training data through a batch upload to `tkyoDriftSetTraining()`
- Request a CLI print out of the log's summary using `npx tkyoDrift -#` (where # is a number)

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

Results are stored in a single CSV file (`drift_log.csv`) with dynamic headers. Each one-off run appends two rows: one for input and one for output. Keep in mind that training data is not added to the log, as there is no drift to compare against for training data.

### Format

```
ID, DEPTH, TIMESTAMP, I/O TYPE, SEMANTIC ROLLING COS, SEMANTIC TRAINING COS, CONCEPT ROLLING COS...
```
- Cosine similarities are recorded per model and baseline type.
- Additional metadata like `depth` and UUIDs are included for tracking.
- UUIDs are shared for the input and output, so that you can review your input/output data logs to inspect which I/O pairs are causing drift.
- Neither the log, nor the binary files, contain your users input or ai outputs.

```
Note: if you add or remove model types to the tkyoDrift tracker, the log will break. Please ensure you clear any existing logs after altering the embedding models.
```

## CLI Tools

### `printLogCLI.js`

Parses `drift_log.csv` and displays violation counts and average cosine similarities over a selected number of days. Uses a color-coded table (green/yellow/red) to show severity of drift. Thresholds are set in this file, and should be adjust to your expected precision needs.

# Architecture

### Embedding Models

TKYO Drift uses remote embedding models on HuggingFace.co for inference. By default, the system operates in JavaScript using a lightweight transformer pipeline, with Python scripts injected as required to improve speed and  enable additional model types.

- `all-MiniLM-L12-v2`: Used for semantic drift or changes in tone or communication style.
- `e5-base`: Used for concept drift or changes in topic or intent.
- `all-MiniLM-L6-v2`: Used for lexical drift or changes in word choice.

Note that while L6 is a subset of L12, it is also the case that lexical drift is a subset of semantic drift. This model can be disabled if you believe that MiniLM-L12 is comprehensive enough to provide drift tracking for both axis.

### Hugging Face Transformer Library

Hugging Face Transformers was chosen as the preferred choice of transformer libraries because:

- We needed access to a broader range of model architectures or domain-specific variants.
- Performance is critical and we wanted to leverage GPU acceleration or quantized models.
- We're working in Python and already rely on Torch, TensorFlow, and JAX-based tooling.
- We need support for advanced features like fine-tuning, token classification, or conditional generation beyond embedding.

Regardless, should you choose to, you may replace the Hugging Face library with Xenova because:
- It runs entirely in Node.js with no backend server or Python runtime.
- Models are executed locally and cached to avoid repeated loads.
- Ideal for CLI tools or small NPM modules where dependencies need to be minimal.

### Drift Types

Drift is detected across combinations of:

- `modelType`: semantic, concept, lexical
- `ioType`: input, output
- `baselineType`: rolling, training

This results in either eight or sixteen cosine similarity comparisons per I/O pair, depending on whether or not training data was provided.
- All embeddings are pooled across all input tokens to get mean values from the input and normalized for all models except for the conceptual drift type.
- Models are loaded sequentially and cached globally to reduce loading time.

Note that once the execution context window closes for the processing I/O pair, models are naturally unloaded. Unless you feel inclined to download and store the models locally in your production pipeline, this is a necessary and unavoidable ~200ms workflow speed penalty.

### Baseline Types

- `Rolling`: A sliding baseline using the most recent N (default 1,000) examples.

The rolling baseline represents the accumulation of inputs and outputs as they are triggered by the production pipeline. As I/O pairs are dispatched to the drift analysis workflow, each I/O pair is saved to the rolling baseline file. When drift calculations are generated, they exclude new I/O pair for that individual calculation but will include them in all subsequent operations. The total number of I/O pairs to be included in drift analysis is a configurable global variable located in the tkyoDrift.js main file. 

- `Training`: A fixed baseline built once from a full dataset.

The training baseline represents the set of inputs used to generate the initial model AI responses, which are ingested by running the 'tkyoDriftSetTraining.js' script. This script performs a batch analysis of each I/O pair used in the training set and creates an artificially 'locked' training file that can not/will not be be appended to over time. Theoretically, you should only need to update this when you have retrained you model.

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

#### `.npy` Rejected

`.npy` was ruled out due to poor support for appending and write concurrency. `.bin` files offer consistent format and linear write performance with no sidecars.

#### No JSON or `.jsonl`

JSON-based storage was removed to reduce size and parsing cost. Embeddings are stored as raw bytes.

### IO Write/Read Methods

File writing/reading is performed using `fs.promises.writeStream` and `fs.promises.readStream`due to the obscene number of floats we need to read, parse and then calculate against, which necessitates that vectors be constructed during the read process to avoid memory overflows. This is an intentional tradeoff of speed for deployability. If you know your system has the memory to spare, IO ops can be improved by replacing write/read streams with loads-to-memory.

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

