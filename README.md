<p align="center">
  <img src="tkyo-banner.png" width="100%" alt="TKYO Drift Banner">
</p>

# AI Temporal Knowledge Yield Output Drift Tracker (TKYO Drift) 

![npm](https://img.shields.io/npm/v/tkyoDrift?style=flat-square)
![license](https://img.shields.io/github/license/oslabs-beta/tkyo-drift?style=flat-square)
![issues](https://img.shields.io/github/issues/oslabs-beta/tkyo-drift?style=flat-square)
![last commit](https://img.shields.io/github/last-commit/oslabs-beta/tkyo-drift?style=flat-square)

TKYO Drift is a lightweight, transparent drift tracking library for AI workflows. It embeds I/O Pairs and compares them to a configurable baseline to detect drift in semantic, conceptual, or lexical meaning over time.

**At the time of writing, this tool is only able to ingest data from mono modal text generation AI workflows.**  

## Overview

This tool is designed for performance-critical environments and was designed to be operable in production environments. The tkyoDrift() function can be inserted into a text generation workflow as an asynchronous function to minimize performance impacts in production settings. It operates quasi-locally with Python dependencies using the a JS transformers library to support both real-time logging of new I/O data, and batch ingestion of I/O training data. Drift is tracked via cosine similarity against baselines, and stored in compact binary files.

## Table of Contents
- [Overview](#overview)
- [Drift Analysis Flow](#drift-analysis-flow)
- [Installation](#how-do-you-install-this-thing)
- [Usage](#how-do-you-use-this-thing)
- [One-off Ingestion](#one-off-ingestion)
- [Training Ingestion](#training-ingestion)
- [Logging](#logging)
- [CLI Tools](#cli-tools)
- [Architecture](#architecture)
- [The Math](#the-math)
- [Scalar Metrics](#scalar-metrics)
- [Future Iterations](#future-iterations)

## Drift Analysis Flow

For each individual input and output, the following workflow is executed in roughly 800ms. 

1. Generate embeddings for both input and output.
2. Save them to `.bin` files (except for training mode).
3. Load historical vectors from disk.
4. Compute baseline vectors as an average of past embeddings.
5. Calculate cosine similarity.
6. Calculate euclidean distance.
7. Capture scalar metrics.
6. Append results to `COS_log.csv`, `EUC_log.csv`, and scalar metric files.

```
Note that the size of input/output text, embedding dimensions, and how many embedding models are chosen will influence the speed of the workflow. Regardless, if tkyoDrift() is called asynchronously, it should not impact your workflow unless you expect a sustained high volume of user inputs per second.
```
# How do you install this thing?
TODO: Update this when the NPM package is built.

1. Install the NPM package:
```bash
npm install tkyoDrift
```
2. Import tkyoDrift into your AI workflow pages:
```js
import tkyoDrift from tkyoDrift
```
3. Add an async function call to the tkyoDrift main function passing in your I/O pair:
```js
...
async tkyoDrift(userinput, aioutput)
... 
```
4. Enjoy the benefits of having drift detection:
```
🏎️☁️☁️☁️ <- THIS GUY IS DRIFTING
```

# How do you use this thing?

You can interact with this library in 4 ways; 
- Dispatch a one-off I/O pair to `tkyoDrift()`
- Dispatch sequential training data through a batch upload to `tkyoDriftSetTraining()`*
- Request a CLI print out of the log's summary using `npx tkyoDrift -#` (where # is a number of days) 
- Export the logs into your Data Viz platform

```
* Do this from a strong PC, and then copy your data into the appropriate folders. Due to a number of factors (I/O Length, CUDA access, memory, cpu cores, training data size) this process can take an exceptionally long time to complete.
```

There is also a small training file downloader script in the util folder called downloadTrainingData.py that you can run to grab the training data from hugging face if you happen to be using a model for your workflow from there.

## One-off Ingestion

`tkyoDrift.js(input, output, depth)` handles individual I/O pairs for drift comparison. It:

- Accepts two text strings as `input` and `output` parameters.
- Accepts an optional 3rd command for the depth* of the I/O pair to enable tracking of nested chain of thought I/O pairs.
- Embeds serially using as many models as you specify.
- Appends to the rolling data file (only).
- Reads from the file to load a baseline for comparison.
- Computes the drift score using cosine similarity.
- Assign a unique ID hash to the I/O pair

```
* Depth tracking is useful when you have multiple chain of thought workflows, and want to track drift per depth level. Just pass a depth argument for how nested each I/O is. For example, your model might have Human -> AI -> AI, where the first input and last output would be depth = 0, and the middleware AI's input and output would be depth = 1.
```

## Training Ingestion

`tkyoDriftSetTraining.js(filepath)` handles full dataset ingestion for baseline creation. It:

- Accepts an array of `{ input, output }` objects.
- Embeds each input in chunks of 100, with 8 in parallel for performance, once for each model type.

```
IMPORTANT! The set training data file should ONLY be run when you intend on replacing the existing tkyoDrift training data file set. This command will obliterate any existing tkyoDrift training files. 

<<< There should only be ONE set of ideal embeddings for your training data. >>>
``` 
As an additional note, you will see a console log warning that inputs exceeding 512 tokens will result in indexing errors. This is accounted for in the embedding process, and all inputs over 512 tokens will return the average vector of each 512 token chunk.

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
- Neither the log, nor the binary files, contain your users input or ai outputs. This data is not necessary to calculate drift, and its exclusion is an intentional choice for data privacy.

```
Note: if you add or remove model types to the tkyoDrift tracker, the log will break. Please ensure you clear any existing logs after altering the embedding model names. What we mean here, is that if you change your lexical model from "lexical" to "linguistic" when writing to the log, the makeLogEntry method of the Drift Class would work, but the log Parser would fail. 

Keep in mind, however, you can change models any time you like, though that will brick drift calculations for a different reason; your inputs/outputs will be embedded with dissimilar methods, which would lead to inaccurate drift calculations.
```

## CLI Tools

TODO: Add the command we need people to enter to trigger this log here after the NPM package is built.

### `printLogCLI.js`

![An image of the print log CLI tool's output, displaying the average cosine similarity across a range of input types, drift types, and baseline types. For example, the input IO Type for the rolling baseline data for the semantic embedding model displays an average cosine similarity of 1, or no semantic drift. ](printLogCLI.png)

Parses `COS_log.csv` and displays violation counts and average cosine similarities over a selected number of days. Uses a color-coded table (green/yellow/red) to show severity of drift. Thresholds are set in this file, and should be adjusted to your expected precision needs.

```
Note: The first record you enter into this system will always show that there is 0 drift when compared against the rolling data set. This is because the rolling dataset will be compared against itself at that point and there will be no drift to detect. This is a known issue, and was an intentional choice as the alternative would be to exclude a write to the `COS_log.csv` and `EUC_log.csv` logs on first write. 

If this bothers you, you can remove line 2 from the COS and EUC logs after you use this system at least twice.
```

### `printScalarCLI.js`

![An image of the print Scalar CLI tool's output, displaying metric comparisons between the training and rolling baseline values. For example, avgWordLength for inputs has a mean of 4.82 characters in the training data, but a mean of 4.49 characters in the rolling data, leading to a mean delta of -0.34, and a PSI value of 0.017 which represents stable populations between the two. ](printScalarCLI.png)

This tool parses the scalar jsonl files to calculate scalar distributions across the training and rolling datasets and delta mean and delta standard deviation between the two distributions. Uses a color-coded table (green/yellow/red) to show severity of drift. Thresholds are set in this file, and should be adjusted to your expected precision needs.

The scalar metrics the system is currently tracking are listed below:

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
Note: Without batch embedding your training data, you will be unable to view scalar metrics. However, we will still collect scalar metrics for your rolling data in the event you add training data at a later time.
```

# Architecture

### Embedding Models

TKYO Drift uses remote embedding models on HuggingFace.co for inference using the Xenova Transformers library in javascript or the python native hugging face library in python. By default, the system operates in JavaScript using a lightweight transformer pipeline, with Python scripts injected as required to improve speed when performing batched operations. The python equivalent embedding pipeline allows for CPU usage by default if your system has the appropriate CUDA drivers.

- `all-MiniLM-L12-v2`: Used for semantic drift or changes in tone or communication style.

  https://huggingface.co/Xenova/all-MiniLM-L12-v2

- `e5-base`: Used for concept drift or changes in topic or intent.

  https://huggingface.co/Xenova/e5-base

- `all-MiniLM-L6-v2`: Used for lexical drift or changes in word choice.

  https://huggingface.co/Xenova/all-MiniLM-L6-v2


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

This results in twelve cosine similarity and euclidean distance comparisons (assuming you use the default models) for each tkyoDrift.js function call.
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

- `Hybrid`: Use the rolling file to provide oldest K (default 10,000) and newest N (default 1,000) simultaneously.

In the event that there is no training data supplied to the system, the Drift analyzer will use the oldest K values entered into the rolling file to represent mock 'training' data, while the newest K values represent the 'rolling' data. This allows us to compare drift against an anchor point to see drift over time, while still having a rolling window to see shock impacts to the system caused by new concepts/semantics/lexicon.


### Binary Embedding Storage

Embeddings are saved in `.bin` files using float32 for efficient storage. This minimizes disk I/O and enables fast appending.

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

***Write operations are done primarily to the rolling file set, as training files are explicitly and intentionally excluded from write operations outside of the batched training embedding pipeline.***

This intentional decision reflects the nature that training datasets represent a fixed point in time, and should not be modified after being ingested. Throughout this codebase, there are checks for a model's baseline type, and if that baseline is set to `training', write operations are skipped.

As a way of making this system work where there is no training data provided, the system will attempt to use hybrid mode as several models use the rolling file path locations as replacements for the training file paths. 

In hybrid mode, because the first N vectors are considered training vectors, and the last K vectors are considered rolling vectors, there will be a duration of time that training and rolling datasets will be equivalents. For example, when the system only contains 1500 vectors, all 1500 will be considered `training` (training defaults are first 10,000) and the most recent 1000 would be considered `rolling`.

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

## Scalar Metrics

In addition to vector-based drift (cosine similarity and Euclidean distance), TKYO Drift also tracks **scalar metrics**—individual numerical features extracted from the raw text of your inputs and outputs. These scalar values help capture shifts in text structure, tone, or complexity that may not be reflected in semantic embeddings.

Scalar metrics are only compared when a **training baseline** is available. Without this baseline, we cannot evaluate how the distribution of scalar values has changed over time.

### How Are They Calculated?

Each metric is computed as follows:

- `norm`: L2 norm (vector magnitude) from the output embedding.
- `textLength`: Length of the raw string.
- `tokenLength`: Number of tokens from the model tokenizer.
- `entropy`: Shannon entropy over character frequencies.
- `avgWordLength`: Mean word length based on whitespace splitting.
- `punctuationDensity`: `punctuation_count / total_chars`.
- `uppercaseRatio`: `uppercase_count / total_chars`.

These are implemented in both JS (for rolling ingestion) and Python (for training ingestion) and saved alongside vector data.

### What Do They Tell Us?

Scalar metrics help detect **non-semantic drift**. For example:

- A spike in `uppercaseRatio` may indicate aggressive or stylized tone.
- A drop in `entropy` could mean the system is outputting simpler or repetitive responses.
- A change in `tokenLength` may point to output truncation or verbosity.

In general, scalar drift reveals changes in *how* your model or users are communicating, not just *what* it's communicating.

### Comparison Logic

Drift is measured by comparing the **mean** and **standard deviation** of each metric between the training and rolling baselines.

For each metric:
```
meanDelta = rollMean - trainMean
stdDelta  = rollStd  - trainStd
```
These deltas are printed in the CLI printout with color-coded thresholds (green/yellow/red) to indicate severity.

### PSI Values (Population Stability Index)

TKYO Drift calculates the **Population Stability Index (PSI)** for all scalar metrics when comparing rolling data against the training baseline. PSI quantifies how much a distribution has shifted over time, and is commonly used in production monitoring to detect silent model degradation.

For each scalar metric, the PSI score is computed using pre-defined bins and normalized frequency distributions from both the rolling and training datasets.

The PSI interpretation follows standard thresholds:

| PSI Value | Interpretation                  |
|-----------|---------------------------------|
| < 0.1     | No significant drift            |
| 0.1–0.25  | Moderate shift; monitor closely |
| > 0.25    | Major shift; likely model drift |

These scores are included in the CLI drift output and provide a statistically grounded way to monitor changes in features like entropy, token length, or word complexity — even if the semantic embeddings remain stable.

TKYO Drift uses this in tandem with mean and standard deviation deltas to give a robust picture of scalar drift.


# Future Iterations

For this project there are a number of ways this platform could be made better, including:

### Rolling Data Selection

The rolling file currently uses a fixed number of records when loaded during the readFromBin method on the DriftModel class, this fixed number may not be a flexible enough method of selecting the number of records to compare for rolling drift. It would be an improvement to modify this so that it uses entries from the last N days instead. 

Building this would require adding a date to the binary file write method to store when a file was added, or replacing the binary file writer entirely and switching to a different file storage format. Regardless, at the time of this readme writeup, there is no date associated with stored vectors and so there is no way to have the rolling file use the last N days of records instead of a fixed count.

### Picking a better K value

The current implementation sets K using a heuristic: K = int(sqrt(num_vectors / 2)), which balances clustering granularity with speed. While the elbow method offers a more statistically grounded way to choose K by evaluating clustering performance across several K values, it is computationally intensive.

Implementing the elbow method would require running KMeans multiple times and analyzing metrics like SSE or silhouette scores. Given our real-time and batch constraints, we avoid this due to diminishing accuracy gains (often logarithmic) versus increased computational cost (often linear to exponential with larger datasets).

### Depth Tracking

While being able to track multiple chains of thought in your AI workflow is an important part of measuring drift, the current implementation of depth tracking suffers from a critical drawback in that COS similarity, EUC distance, and scalar metrics are NOT calculated for each depth band. 

Resolving this would require a major refactor of the calculation of drift metrics so that they only compare against other entries of the same depth. Alternatively, it would require writing to individual logs for each depth. In any case, the current functionality may unintentionally pollute the drift scores as I/O from different depths are included in the calculations for COS similarity/EUC distance/Scalar metrics.

### Python vs Javascript

This project was initially built as a pure javascript project to enable wider deployment, but various functions and libraries were originally built, and intended to be used in python. As a result, this project was refactored after an initial test build to include a javascript pipeline for individual embeddings and a python version for batched embeddings.

What this means is that the tkyoDriftSetTraining.py file and the tkyoDrift.js processes are functionally duplicates of each other except that the former is explicitly meant to be called once for a batch, while the later is meant to be invoked on every new input.

This is fine as it is, but since many javascript libraries are just python scripts wearing a disguise, it would be ideal to rebuild this entire platform in python with a javascript NPM package to install it, and a javascript function hook to pass data into it. This would allow this system to avoid unnecessary conversion from javascript into python to execute AI embeddings, calculate K means, or generate the HNSW index.

### PSI Logging

At the time of writing, this project does not log PSI values to a csv file. This means that while PSI values are calculated on demand using the `printScalarCLI.js`, there are no exportable scalar metrics for external data visualization tools.

Fixing this would involve adding a cronjob to compare scalar metrics on an interval, or adding a counter of some sort and triggering a scalar comparison every N (10? 100?) new inputs. If the output of this comparison is sent to a log instead of the CLI, it would be consumable by external tools.

### Cloud Based embedding services

Waiting is pain, and embedding hundreds of thousands of inputs over and over again can take a long time. Not to mention that larger models take up a ton of space (1 mill I/Os is like 20 gbs). This whole platform could be a paid service where people upload their I/Os and you keep their embeddings remotely.

Not only that, but there is a vast range of data visualizations that could be made, warning and alerts, recommendations based on what flags are getting triggered, etc. 

This would involve creating a whole front end with user login, a backend API to receive one off calls and a file upload system to receive massive training data files. This would be a fun project in it's own right, but obviously involves cloud server costs to rapidly process embeddings. If you do decide to make a business out of this, give us a call, we would love to help. 

### Multi-Modality

This project is only developed to be able to ingest text data, but there are many AI workflows out there. A future iteration of this project could include text2img, img2img, text2video, img2video, etc. 

Most of the ground work for this is already done, but new embedding models specifically designed for those types of workflows would need to be incorporated, along with file type handling which is missing in the main workflow. (If you pass anything other than text into the main or batch embedding function, they break.)

### More Math

We are currently calculating a number of metrics from both individual vectors as well as populations across the training and rolling data sets. There is room for improvement, however, in that there are additional drift measures such as KL divergence and Earth Movers distance that would be useful to calculate in this workflow. 

It is our opinion that either of these could be added to the tkyoDrift analysis with minimal effort, as the `tkyoDrift` main logic `tkyoDriftSetTraining.py` batch embedding logic both capture scalar metrics for population comparisons already.

## Contributing

We welcome contributions, ideas, and pull requests!  
If you’d like to improve TKYO Drift, feel free to fork the repo and submit a PR.

Before getting started, check out any open issues and see if you can help.  
If you'd like to propose a feature, feel free to open a discussion or ticket.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Built with blood sweat and tears by the TKYO Drift team:

Milti, Wing, Monique, Chris and Anthony

Check out our other projects and send us a message if you would like to collaborate.

Big thanks to the folks behind:
- [Hugging Face Transformers](https://huggingface.co)
- [Xenova Transformers](https://xenova.github.io/)
- [Open Source Labs](https://github.com/oslabs-beta)
- [Codesmith](https://www.codesmith.io/)