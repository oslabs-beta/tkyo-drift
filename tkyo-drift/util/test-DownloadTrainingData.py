# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset

# TODO: Delete this whole file before going live?
data_location = "SmallDoge/SmallThoughts"

def dataSetLoader (data_location):
    dataset = load_dataset("SmallDoge/SmallThoughts")
    print(dataset)
    return dataset

dataSetLoader(data_location)