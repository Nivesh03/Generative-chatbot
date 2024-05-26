from pre import input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, max_encoder_seq_length
from training import encoder_inputs, decoder_inputs, encoder_states, decoder_lstm, decoder_dense, \
encoder_input_data, num_decoder_tokens, num_encoder_tokens, latent_dim

from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np

training_model = load_model('training_model.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.0

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += ' ' + sampled_token

        if (sampled_token == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence


# for seq_index in range(200, 300):
#     test_input = encoder_input_data[seq_index : seq_index + 1]
#     decoded_sentence = decode_sequence(test_input)
#     print('-')
#     print('Input sentence: ', input_docs[seq_index])
#     print('Decoded sentence: ', decoded_sentence)
        