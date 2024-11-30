import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] ='3'
import random

def load_doc(path : str) -> list:
    """
    Loading the translation file
    
    Inputs:
        path: the path to the data
    
    Outputs:
        lines: returns a list of 'English\tSpanish' pairs
    
    """
    with open(path, encoding='utf-8') as line:
        lines = line.read().split("\n")[:-1]

    #print(type(lines))
    #print(lines)        
    return lines


def make_pairs(data : list) -> list:
    """
    Making English and Spanish pairs in [(,), (,), ...] format

    Inputs:
        data: a list of English\tSpanish sentences

    Outputs:
        pairs: a list of English, Spanish tuples
    """
    pairs = []

    for i in range(len(data)):
        eng, spa = data[i].split("\t")
        spa = "[START] " + spa + " [END]"  # make sure you put a whitespace after [START] and before [END]
        pairs.append((eng, spa))

    return pairs


def train_test_split_data(dataset: list) -> list:
    """
    Splitting data into train, test, and validation sets by 70%, 15%, and 15%

    Input:
        dataset: list of English, Spanish pairs

    Output:
        train, validation, test: 3 lists of train, validation, and test sets    
    """
    random.shuffle(dataset) # to make sure various data lengths are distributed over all sets
    data_length = len(dataset)
    train_size = int(data_length * 0.7)
    validation_size = int((data_length - train_size) * 0.15)
    train = dataset[:train_size]
    validation = dataset[train_size:train_size + validation_size]
    test = dataset[train_size + validation_size:]

    return train, validation, test




