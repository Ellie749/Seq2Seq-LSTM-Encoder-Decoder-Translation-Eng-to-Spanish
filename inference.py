import os
os.environ['TF_MIN_LOG_LEVEL'] = "3"
from tensorflow.keras. models import load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Bidirectional, TimeDistributed, Dropout, GRU
from tensorflow.keras import Model
from tensorflow import add
import numpy as np


def load_encoder_decoder_inference(vocab_size: int, latent_dim: int, embed_dim: int) -> Model:
    """
    Reloads models weights and redefines model architecture for inference purposes

    Inputs:
        vocab_size: maximum number of words in the defined dictionary
        latent_dim: hidden state dimension
        embed_dim: words embedding dimension

    Outputs:
        encode and decoder parts of the architecture
    
    """
    #Encoder
    eng_input = Input(shape=(None,), name="English")
    e_input = Embedding(vocab_size, embed_dim, mask_zero=True)(eng_input)
    _, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(latent_dim, return_sequences=False, return_state=True), merge_mode="sum")(e_input)
    cell_state = add(forward_c, backward_c)
    hidden_state = add(forward_h, backward_h)
    encoder = Model(eng_input, [hidden_state, cell_state])

    #Decoder
    spa_input = Input(shape=(1,), name="Spanish")
    h_input = Input(shape=(latent_dim,), name="hidden_state_initial")
    c_input = Input(shape=(latent_dim,), name="cell_state_initial")
    e_output = Embedding(vocab_size, embed_dim, mask_zero=True)(spa_input)
    y, h, c = LSTM(latent_dim, return_sequences=True, return_state=True)(e_output, initial_state=[h_input, c_input])

    drop = TimeDistributed(Dropout(0.5))(y)
    den = TimeDistributed(Dense(vocab_size, activation='softmax'))(drop)

    decoder = Model([spa_input, h_input, c_input], [den, h, c])

    return encoder, decoder


def translate(encoder: Model, decoder: Model, data, end_token: int, start_token:int) -> np.array:
    """
    Based on encoder and decoder models, it produces a sequence of translated sentence.

    Inputs:
        encoder
        decoder
        data: English sentence to be translated (a tensor of integer numbers)
        end_token: The vectorized version of [END] token which is usually 4
        start_token: The vectorized version of [START] token which is usually 3

    Outputs:
        decoded_sentence: model's Spanish prediction. A numpy array of integers representing words.
    
    """
    decoded_sentence = []
    encoder.load_weights("Best_Model.h5", by_name=True)
    decoder.load_weights("Best_Model.h5", by_name=True)

    en_h, en_c = encoder.predict(np.array([data]))

    en_h = np.array(en_h)
    en_c = np.array(en_c)

    target_seq = np.array([start_token])

    #print(target_seq)

    for i in range(20):        

        de_out, en_h, en_c = decoder.predict([target_seq, en_h, en_c])
        vocab = np.argmax(de_out[0, -1, :])
        
        if(vocab == end_token):
            break
        
        decoded_sentence.append(vocab)
        target_seq = np.array([vocab])
      
    return decoded_sentence