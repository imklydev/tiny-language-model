import model_variables

import torch

import string

# index 0 is reserved for unknown characters
UNK = "\x00"
VOCAB = (
    [UNK]
    + list(string.ascii_lowercase)
    + list(string.ascii_uppercase)
    + list(string.digits)
    + list(" \n")
    + list(".,!?;:'\"-()")
)

CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}

model_variables.VOCABULARY_SIZE = len(VOCAB)


def tokenize_char(c):
    return CHAR_TO_ID.get(c, 0)

def detokenize_token(i):
    return ID_TO_CHAR.get(i, UNK)

def detokenize_tensor_to_string(tensor):
    """
    Returns a string from detokenizing a tensor containing tokens
    """

    tensor = "".join(detokenize_token(token) for token in tensor.tolist())

    return tensor

def tokenize_string(text):
    """
    Returns a tokenized tensor
    """

    
    tokenized_data = []

    for c in text:
        tokenized_data.append(tokenize_char(c))

    tokenized_tensor = torch.tensor(tokenized_data, dtype=torch.float32)

    return tokenized_tensor

def tokenize(data):
    """
    Returns a tokenized list
    """

    tokenized_data = []

    for text in data:
        for c in text:
            tokenized_data.append(tokenize_char(c))

    return tokenized_data