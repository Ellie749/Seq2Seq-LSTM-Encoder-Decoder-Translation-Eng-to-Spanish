import string
import re
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow import strings


def custom_standardize(target_data):
    """
    Standardizes a list of strings by removing all the standard punctuation plus "¿" and "¡" but keeps [ and ] because of [START] and [END].
    Lowercases and removes whitespaces from begginning and the end of the strings.
    By keeping [] it might take each as a token but given the text itself it is not a common token except from [START] and [END].

    Input:
        target_data: a list of strings to be stardardized but the type is <class 'tensorflow.python.framework.ops.EagerTensor'>

    Output:
        target_data_result: standardized list of strings
    """
    puncs = string.punctuation
    puncs = puncs + "¿" + "¡"
    puncs = puncs.replace("[", "")
    puncs = puncs.replace("]", "")
    print(type(target_data))

    '''
    target_data = target_data.lower()
    target_data = target_data.strip()
    target_data_result = ""
    for i in target_data:
       if i not in puncs:
           target_data_result += i
    return target_data_result
    '''
    target_data_processed = strings.lower(target_data)
    target_data_processed = strings.strip(target_data_processed)

    return strings.regex_replace(
        target_data_processed, f"[{re.escape(puncs)}]", "")
    

def slicing(data: list):
    eng = [sentence[0] for sentence in data]
    spa = [sentence[1] for sentence in data]

    return eng, spa

def vectorize(train: list, validation: list, test: list, max_token: int, sequence_length: int, batch_size: int) -> tf.data.Dataset:
    """
    Vectorizing source and target sentences.

    Input:
        data: a list of tuples that contains source and target strings

    Output:

    """
    eng_train, spa_train = slicing(train)
    eng_val, spa_val = slicing(validation)
    eng_test, spa_test = slicing(test)

    #print(spa_test)
    eng_vectorization = TextVectorization(
        max_tokens = max_token,
        output_mode = 'int',
        output_sequence_length = sequence_length,
        standardize = 'lower_and_strip_punctuation'
    )
    #You can define different max_token and sequence_length for eng and spa
    spa_vectorization = TextVectorization(
        max_tokens = max_token,
        output_mode = 'int',
       output_sequence_length = sequence_length + 1,
        standardize = custom_standardize
    )

    train_data = tf.data.Dataset.from_tensor_slices((eng_train, spa_train))
    train_data = train_data.batch(batch_size)

    validation_data = tf.data.Dataset.from_tensor_slices((eng_val, spa_val))
    validation_data = validation_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((eng_test, spa_test))
    test_data = test_data.batch(batch_size)
    
    eng_vectorization.adapt(eng_train)
    spa_vectorization.adapt(spa_train)

    train_data = train_data.map(lambda x, y: ({"English": eng_vectorization(x), "Spanish": spa_vectorization(y)[:, :-1]}, spa_vectorization(y)[:, 1:]), num_parallel_calls=4)
    validation_data = validation_data.map(lambda x, y: ({"English": eng_vectorization(x), "Spanish": spa_vectorization(y)[:, :-1]}, spa_vectorization(y)[:, 1:]), num_parallel_calls=4)
    test_data = test_data.map(lambda x, y: ({"English": eng_vectorization(x)}, y), num_parallel_calls=4)

    return train_data, validation_data, test_data, spa_vectorization


    
