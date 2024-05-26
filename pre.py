import numpy as np
import re
import pandas as pd

data_path = 'dialogs.txt'

# Read and split lines from the file
with open(data_path, 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')

# Function to clean and tokenize text
def clean_text(text):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and clean punctuation
    tokens = re.findall(r"[\w']+|[^\w\s]", text)
    tokens = [token for token in tokens if token.isalpha() or token in ["<START>", "<END>"]] 
    return ' '.join(tokens)

# Initialize lists and sets
input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for line in lines[:3000]:
    input_doc, target_doc = line.split('\t')
    input_doc = clean_text(input_doc)
    target_doc = '<START> ' + clean_text(target_doc) + ' <END>'
    input_docs.append(input_doc)
    target_docs.append(target_doc)

    for token in input_doc.split():
        input_tokens.add(token)
    for token in target_doc.split():
        target_tokens.add(token)

input_tokens = sorted(input_tokens)
target_tokens = sorted(target_tokens)

input_features_dict = {token: index for index, token in enumerate(input_tokens)}
target_features_dict = {token: index for index, token in enumerate(target_tokens)}

reverse_input_features_dict = {index: token for token, index in input_features_dict.items()}
reverse_target_features_dict = {index: token for token, index in target_features_dict.items()}

num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)
max_encoder_seq_length = max([len(doc.split()) for doc in input_docs])
max_decoder_seq_length = max([len(doc.split()) for doc in target_docs])

encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for t, token in enumerate(input_doc.split()):
        encoder_input_data[i, t, input_features_dict[token]] = 1.0
    for t, token in enumerate(target_doc.split()):
        decoder_input_data[i, t, target_features_dict[token]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_features_dict[token]] = 1.0

# Your cleaned and processed data is now ready for use

# print(input_tokens)
# print(target_tokens)
# print(input_docs)
# print()
# print(tuple(zip(input_docs[20:40], target_docs[20:40])))