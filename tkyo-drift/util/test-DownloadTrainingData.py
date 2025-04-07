# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset

# TODO: Delete this whole file before going live?
data_location = "SmallDoge/SmallThoughts"
# data_location = 'open-thoughts/OpenThoughts2-1M'

def dataSetLoader (data_location):
    dataset = load_dataset("SmallDoge/SmallThoughts")
    # dataset = load_dataset("open-thoughts/OpenThoughts2-1M")
    print(dataset)
    return dataset

dataSetLoader(data_location)