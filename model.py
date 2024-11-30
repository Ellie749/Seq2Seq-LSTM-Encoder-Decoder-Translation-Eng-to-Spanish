import os
os.environ["TF_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Bidirectional, TimeDistributed, Dropout, GRU
from tensorflow.keras import Model
from tensorflow import add
from tensorflow.keras.callbacks import ModelCheckpoint

'''
class model(Model):
    def __init__(self, vocab_size, latent_dim, embed_dim):
        super(model, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
'''
def build_model(vocab_size, latent_dim, embed_dim):
    """
    Seq2seq model architecture for training

    Inputs:
        vocab_size: maximum number of words in the defined dictionary
        latent_dim: hidden state dimension
        embed_dim: words embedding dimension

    Outputs:
        model: an encoder-decoder model for seq2seq prediction
    
    """
    eng_input = Input(shape=(None,), name="English") # name should be the same as the tag of dictionary 
    #If you don't specify name for an Input layer, Keras will generate a default name such as input_1, input_2, etc., based on the order of the input layer creation.
    e_input = Embedding(vocab_size, embed_dim, mask_zero=True)(eng_input)

    spa_input = Input(shape=(None,), name="Spanish")
    e_output = Embedding(vocab_size, embed_dim, mask_zero=True)(spa_input)

    output, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(latent_dim, return_state=True), merge_mode="sum")(e_input) #GRU returns 3 outputs. no cell states
    # If return_sequences == False -> output == forwardh + backwardh
    # since we are specifying the initial state, both h and c should be stated and not just one of them. 
    # If we want cell to be random, we can write random function manually
    
    cell_state = add(forward_c, backward_c)
    y = LSTM(latent_dim, return_sequences=True)(e_output, initial_state=[output, cell_state])

    drop = TimeDistributed(Dropout(0.5))(y)
    den = TimeDistributed(Dense(vocab_size, activation='softmax'))(drop)

    model = Model([eng_input, spa_input], den)

    return model
   

def train(seq2seq, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS):
    """
    train and save model
    
    Inputs:
        seq2seq: model architecture
        train_dataset: train data
        validation_dataset: validation data
        BATCH_SIZE: batch size
        EPOCHS: epochs

    Outputs:
        H: history of training

    """
    seq2seq.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    checkpoint = ModelCheckpoint('Best_Model.h5', monitor='val_accuracy', mode='max', save_best_only='True', save_format="h5")
    H = seq2seq.fit(train_dataset, validation_data=validation_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[checkpoint])
    
    return H
