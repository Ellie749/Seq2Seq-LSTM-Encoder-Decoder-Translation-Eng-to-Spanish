import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import save_model
from tensorflow import config
from tensorflow import slice
import utils
import data_loader
import tokenizer
import model
import inference

#print(config.list_physical_devices("GPU"))
#print(config.list_physical_devices("CPU"))


DATA_PATH = "spa.txt"
MAX_TOKEN = 15000
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
EPOCHS = 15
EMBED_DIM = 256
LATENT_DIM = 1024


def main():
    doc = data_loader.load_doc(DATA_PATH)
    pairs = data_loader.make_pairs(doc)
    train, validation, test = data_loader.train_test_split_data(pairs)
    #print(test)
    train_dataset, validation_dataset, test_dataset, spanish_tokenizer = tokenizer.vectorize(train, validation, test, MAX_TOKEN, SEQUENCE_LENGTH, BATCH_SIZE)
    vocabs = spanish_tokenizer.get_vocabulary() # to reverse predictions from numbers to words

    """
    print(len(train))
    for inputs, targets in train_dataset.take(1):
        print(f"english input: {inputs['English'].shape}")
        print(f"Spanish input: {inputs['Spanish'].shape}")
        print(f"Spanish output: {targets.shape}")
   
    for inputs in test_dataset.take(1):
        print(f"english test input: {inputs['English'].shape}")

    Run this for train
    seq2seq = model.build_model(MAX_TOKEN, LATENT_DIM, EMBED_DIM)
    H = model.train(seq2seq, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS)
    utils.plot_metrics(H)
    """

    #Take sample data for testing
    for inputs in test_dataset.take(1):
        t = inputs[0]["English"][0]
        label = inputs[1]
 
    encoder, decoder = inference.load_encoder_decoder_inference(MAX_TOKEN, LATENT_DIM, EMBED_DIM)
 
    end_token = vocabs.index("[end]") #int
    start_token = vocabs.index("[start]") #int

    translated_seq = inference.translate(encoder, decoder, t, end_token, start_token)
    #print(translated_seq)
    #print(type(translated_seq))

    final_translation = []
    
    for i in range(len(translated_seq)):
        final_translation.append(vocabs[translated_seq[i]])
    
    print("Prediction: ", " ".join(final_translation))
    print("Correct Translation: ", label[0].numpy().decode('utf-8'))

if __name__ == "__main__":
    main()
