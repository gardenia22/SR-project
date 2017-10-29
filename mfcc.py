__author__ = 'gardenia'
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc

def audioToInputVector(audio, fs, numcep, numcontext, fixedlength):

    # Get mfcc coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep)

    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*numcontext+1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1])
        )

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    # Copy the strided array so that we can write to it safely
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

    # limit inputs within fixedlength, padding 0

    padding_shape = (max(0, fixedlength-train_inputs.shape[0]), train_inputs.shape[1])
    train_inputs = np.vstack([train_inputs[:fixedlength],
                             np.zeros(padding_shape)])

    # Return results
    return train_inputs


def audiofile_to_input_vector(audio_filename, numcep, numcontext, fixedlength):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    return audioToInputVector(audio, fs, numcep, numcontext, fixedlength)

def load_inputs(path):
    # load dataset from s1/ director, convert into mfcc features with fixedlength
    import os
    import fnmatch
    source_dir = path
    X = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.wav"):
            wav_file = os.path.join(root, filename)
            numcep = 12
            numcontext = 0
            fixedlength = 230
            features =  audiofile_to_input_vector(wav_file, numcep, numcontext, fixedlength)
            X.append(features)
    X = np.array(X)
    return X

if __name__=="__main__":
    X = load_inputs("s1/")
    np.save("audio_inputs.npy", X)
    print "load training inputs of dimension", np.array(X).shape