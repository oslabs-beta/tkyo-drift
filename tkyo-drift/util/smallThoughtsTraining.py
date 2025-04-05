from datasets import load_dataset

data_location = "SmallDoge/SmallThoughts"

def dataSetLoader (data_location):
    dataset = load_dataset("SmallDoge/SmallThoughts")
    print(dataset)
    return dataset

dataSetLoader(data_location)