

def get_sequence_length(train_inputs):
    '''
    Arg: 
        train_inputs of shape [num_input, max_length, num_features]
    Return: 
        array of sequence lengths of shape [num_input, 1]
    '''
    max_length = train_inputs.shape[1]
    num_inputs = train_inputs.shape[0]
    result = []
    for i in range(num_inputs):
        result.append(max_length)
    return np.array(result, dtype=np.int64)

def sparse_tuples_from_sequences(sequences, dtype=np.int32):
    """
    Create a sparse representations of inputs.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indexes = []
    values = []

    for n, sequence in enumerate(sequences):
        indexes.extend(zip([n] * len(sequence), range(len(sequence))))
        values.extend(sequence)

    indexes = np.asarray(indexes, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indexes).max(0)[1] + 1], dtype=np.int64)

    return indexes, values, shape


def sequence_decoder(sequence, first_index=(ord('a') - 1)):
    """
    Decode text files.
    Args:
        sequence: list of int.
            Encoded sequence
        first_index: int.
            First index (usually index of 'a').
    Returns:
        decoded_text: string.
    """
    decoded_text = ''.join([chr(x) for x in np.asarray(sequence) + first_index])
    # Replacing blank label to none.
    decoded_text = decoded_text.replace(chr(ord('z') + 1), '')
    # Replacing space label to space.
    decoded_text = decoded_text.replace(chr(ord('a') - 1), ' ')
    return decoded_text


def texts_encoder(texts, first_index=(ord('a') - 1), space_index=0, space_token='<space>'):
    """
    Encode texts to numbers.
    Args:
        texts: list of texts.
            Data directory.
        first_index: int.
            First index (usually index of 'a').
        space_index: int.
            Index of 'space'.
        space_token: string.
            'space' representation.
    Returns:
        array of encoded texts.
    """
    result = []
    for text in texts:
        item = make_char_array(text, space_token)
        item = np.asarray([space_index if x == space_token else ord(x) - first_index for x in item])
        result.append(item)

    return np.array(result)

