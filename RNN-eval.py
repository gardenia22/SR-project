import time
import sys
import math
import numpy as np
import tensorflow as tf
from utils import *
import pickle



# Data directories. TODO
DATA_DIR = "./data/"
pred = True
LOAD_MODEL_NAME = "RNN_model/autoencoder/1000_epochs_20_batchsize/model_600.ckpt"
# LOAD_MODEL_NAME = "RNN_model/autoencoder/1000_epochs_20_batchsize/model_400.ckpt"
# LOAD_MODEL_NAME = "RNN_model/autoencoder/500_epochs_20_batchsize/model_complete.ckpt"



# LOAD_MODEL_NAME = "RNN_model/mfcc_best/model.ckpt_complete"

LSTM = True
# Constants.
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


# Accounting the 0th index + space + blank label = 28 characters
NUM_CLASSES = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters.
NUM_EPOCHS = 50
NUM_HIDDEN = 50
NUM_LAYERS = 1
BATCH_SIZE = 30

# Optimizer parameters.
INITIAL_LEARNING_RATE = 1e-3
MOMENTUM = 0.9

TEST_SIZE_RATIO = 0.1



def main(argv):
    # arguement option: mfcc and autoencoder 
   
    if argv[1] == 'mfcc':
        # inputs shape [num_samples, max_length, num_features]
        # sequence_length shape [num_samples,]
        inputs = np.load(DATA_DIR + "mfcc/inputs.npy")
        RESULT_PATH = "./RNN_result/mfcc/" 
        MODEL_NAME = "./RNN_model/mfcc/model"
    else: 
        inputs = np.load(DATA_DIR + "autoencoder/inputs.npy")
        inputs = inputs[:,:, np.newaxis]
        RESULT_PATH = "./RNN_result/autoencoder/"
        MODEL_NAME = "./RNN_model/autoencoder/model"

    # Split training and testing inputs
    train_inputs = inputs[:int(inputs.shape[0] * (1-TEST_SIZE_RATIO))]
    train_inputs = np.float32(train_inputs) 
    train_sequence_lengths = get_sequence_length(train_inputs)

    test_inputs = inputs[int(inputs.shape[0] * (1-TEST_SIZE_RATIO)):] 
    test_inputs = np.float32(test_inputs) 
    test_sequence_lengths = get_sequence_length(test_inputs)
        
    # read texts 
    texts = read_text_file(DATA_DIR + "align_text")

    train_texts = texts[:int(texts.shape[0] * (1-TEST_SIZE_RATIO))]
    test_texts = texts[int(texts.shape[0] * (1-TEST_SIZE_RATIO)):]


    # read labels
    labels = texts_encoder(texts, first_index=FIRST_INDEX,
                                  space_index=SPACE_INDEX,
                                  space_token=SPACE_TOKEN)

    train_labels = labels[:int(labels.shape[0] * (1-TEST_SIZE_RATIO))]
    test_labels = labels[int(labels.shape[0] * (1-TEST_SIZE_RATIO)):] 



    NUM_FEATURES = train_inputs.shape[2] 


 

    graph = tf.Graph()
    with graph.as_default():

        print ("Building graph...")
        print ("LOAD_MODEL_NAME: ", LOAD_MODEL_NAME)
        
        inputs_placeholder = tf.placeholder(tf.float32, [None, None, NUM_FEATURES])

        # SparseTensor placeholder required by ctc_loss op
        labels_placeholder = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size].
        sequence_length_placeholder = tf.placeholder(tf.int32, [None])


        if LSTM:
            cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
            stack = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS,
                                                    state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(stack, inputs_placeholder, sequence_length_placeholder, dtype=tf.float32)
        else:
            # create a BasicRNNCell
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(NUM_HIDDEN)

            # defining initial state
            initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
            # 'inpus_placeholder' is a tensor of shape [batch_size, max_time, cell_state_size]
            # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs_placeholder, sequence_length=sequence_length_placeholder, initial_state=initial_state, dtype=tf.float32)

      
        shape = tf.shape(inputs_placeholder)
        batch_size, max_time_steps = shape[0], shape[1]


        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])

        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1),name='weights')

        bias = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]),name='bias')


        # Doing the affine projection.
        logits = tf.matmul(outputs, weights) + bias

        # Reshaping back to the original shape.
        logits = tf.reshape(logits, [batch_size, -1, NUM_CLASSES])

        # Time is major.
        logits = tf.transpose(logits, (1, 0, 2))

        # inputs is a tensor of shape [max_time x batch_size x num_classes]
        with tf.name_scope('loss'):
            # loss is a 1-D tensor of shape [batch]
            loss = tf.nn.ctc_loss(labels_placeholder, logits, sequence_length_placeholder)
            cost = tf.reduce_mean(loss)

        optimizer = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, 0.9).minimize(cost)
        #optimizer = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(cost)

        # CTC decoder.
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, sequence_length_placeholder)

        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                           labels_placeholder))


        print ("Finished building graph.")

    with tf.Session(graph=graph) as session:

        print ("Running session...")

        # Saver op to save and restore all the variables.
        saver = tf.train.Saver()
        if pred == True:
            saver.restore(session, LOAD_MODEL_NAME)
 
        test_num = test_inputs.shape[0]
        r = BATCH_SIZE/test_num -1
        test_inputs_pl = test_inputs
        test_sequence_lengths_pl = test_sequence_lengths
        test_labels_pl = test_labels
        for i in range(r):
            test_inputs_pl = np.vstack((test_inputs_pl, test_inputs))
            test_sequence_lengths_pl = np.append(test_sequence_lengths_pl, test_sequence_lengths)
            test_labels_pl = np.append(test_labels_pl, test_labels)

        test_targets = sparse_tuples_from_sequences(test_labels_pl[0:BATCH_SIZE])
        test_feed = {inputs_placeholder: test_inputs_pl[0:BATCH_SIZE],
                     labels_placeholder: test_targets,
                     sequence_length_placeholder: test_sequence_lengths_pl[0:BATCH_SIZE]}
        test_label_error_rate = session.run(label_error_rate, feed_dict=test_feed)
        print "test error rate:", test_label_error_rate

        test_feed = {inputs_placeholder: train_inputs[0:BATCH_SIZE],
                      labels_placeholder: sparse_tuples_from_sequences(train_labels[0:BATCH_SIZE]),
                      sequence_length_placeholder: train_sequence_lengths[0:BATCH_SIZE]
                      }

        # Decoding.
        test_label_error_rate = session.run(label_error_rate, feed_dict=test_feed)
        print "train error rate:", test_label_error_rate
        decoded_outputs = session.run(decoded[0], feed_dict=test_feed)
        dense_decoded = tf.sparse_tensor_to_dense(decoded_outputs, default_value=-1).eval(session=session)

        result = []
        for i, sequence in enumerate(dense_decoded):
            if i >= test_num:
                break
            sequence = [s for s in sequence if s != -1]
            decoded_text = sequence_decoder(sequence)
        
            seq = "Sequence {}/{}\n".format(i + 1, test_num)
            org = "Original:\n{}\n".format(train_texts[i])
            dec = "Decoded:\n{}\n".format(decoded_text)
            # print (seq)
            # print (org)
            # print (dec)
        
            result.append(seq)
            result.append(org)
            result.append(dec)


    
        with open(RESULT_PATH + "testing_result.txt", "w") as f:
            for r in result:
                f.write(r)
        print ("testing result saved in", RESULT_PATH + "testing_result.txt")





       


if __name__ == '__main__':
    tf.app.run()