# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
# Import helper function to load and embed the data
from util import pythonTrainingEmb
from util.writeSharedScalars import write_shared_scalar_metrics


import time

def tkyoDriftSetTraining(data_set_Path, input_name="input", output_name="output"):

    # Starts the total function timer
    startTotal = time.perf_counter()

    # Define the models
    MODELS = {
        "semantic": "sentence-transformers/all-MiniLM-L12-v2",
        "concept": "intfloat/e5-base",
        "lexical": "sentence-transformers/all-MiniLM-L6-v2",
    }

    IO_TYPES = {
        "input": input_name,
        "output": output_name,
    }
    # Call once per I/O type to extract shared scalar metrics
    for io_type, column_name in IO_TYPES.items():
        write_shared_scalar_metrics(data_set_Path, io_type, column_name)

    # Iterate through models dictionary
    for model_type, model_name in MODELS.items():
        for io_type, io_type_name in IO_TYPES.items():
            pythonTrainingEmb.trainingEmb(
                model_type=model_type,
                model_name=model_name,
                data_path=data_set_Path,
                io_type=io_type,
                io_type_name=io_type_name,
            )

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Elapsed: {endTotal - startTotal:.6f} seconds")

    return

# TODO Remove hardcoded path, input name, & output name
DATASET_PATH = "./data"
input_name = "problem"
output_name = "solution"
# input_name = "['conversations'][0]['value']" 
# output_name = "['conversations'][1]['value']"
tkyoDriftSetTraining(DATASET_PATH, input_name, output_name)
