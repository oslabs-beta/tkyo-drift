# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset

# data_location = "SmallDoge/SmallThoughts"
# data_location = 'open-thoughts/OpenThoughts2-1M'
# data_location ='gretelai/synthetic_text_to_sql'
# data_location = 'BatsResearch/bonito-experiment'
data_location = 'TsukiOwO/open-thoughts_OpenThoughts2-1M'

def dataSetLoader (data_location):
    # dataset = load_dataset("SmallDoge/SmallThoughts")
    # dataset = load_dataset("gretelai/synthetic_text_to_sql")
    # dataset = load_dataset("open-thoughts/OpenThoughts2-1M")
    # dataset = load_dataset("BatsResearch/bonito-experiment")
    dataset = load_dataset("TsukiOwO/open-thoughts_OpenThoughts2-1M")
    print(dataset)
    return dataset

dataSetLoader(data_location)