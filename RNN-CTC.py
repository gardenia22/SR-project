import time
import sys
import math
import numpy as np
import tensorflow as tf
from utils import *
import pickle


# MODEL_PATH = "./RNN_model/"
# MODEL_NAME = MODEL_PATH + "model"
# RESULT_PATH = "./RNN_result/"

# Data directories. TODO
DATA_DIR = "./data/"

# Constants.
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


# Accounting the 0th index + space + blank label = 28 characters
NUM_CLASSES = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters.
NUM_EPOCHS = 1000
NUM_HIDDEN = 50
NUM_LAYERS = 1
BATCH_SIZE = 20

# Optimizer parameters.
INITIAL_LEARNING_RATE = 1e-3
MOMENTUM = 0.9

TEST_SIZE_RATIO = 0.1

'''
train 0...0.8
val  0.8---0.9
test 0.9...1

'''

def main(argv):
    # arguement option: mfcc and autoencoder 
   
    if argv[1] == 'mfcc':
        # inputs shape [num_samples, max_length, num_features]
        # sequence_length shape [num_samples,]
        inputs = np.load(DATA_DIR + "mfcc/inputs.npy")
        RESULT_PATH = "./RNN_result/mfcc/" 
        MODEL_NAME = "./RNN_model/mfcc/model.ckpt"
    else: 
        inputs = np.load(DATA_DIR + "autoencoder/inputs.npy")
        inputs = inputs[:,:, np.newaxis]
        RESULT_PATH = "./RNN_result/autoencoder/"
        MODEL_NAME = "./RNN_model/autoencoder/model.ckpt"

    # Split training and testing inputs
    train_inputs = inputs[:int(inputs.shape[0] * (1-2*TEST_SIZE_RATIO))]
    train_inputs = np.float32(train_inputs) 
    train_sequence_lengths = get_sequence_length(train_inputs)

    test_inputs = inputs[int(inputs.shape[0] * (1-TEST_SIZE_RATIO)):] 
    test_inputs = np.float32(test_inputs) 
    test_sequence_lengths = get_sequence_length(test_inputs)

    validation_inputs = inputs[int(inputs.shape[0] * (1-2*TEST_SIZE_RATIO)):int(inputs.shape[0] * (1-TEST_SIZE_RATIO))]
    validation_inputs = np.float32(validation_inputs)
    validation_sequence_lengths = get_sequence_length(validation_inputs)
        
    # read texts 
    texts = read_text_file(DATA_DIR + "align_text")

    train_texts = texts[:int(texts.shape[0] * (1-2*TEST_SIZE_RATIO))]
    test_texts = texts[int(texts.shape[0] * (1-TEST_SIZE_RATIO)):]
    validation_texts = texts[int(texts.shape[0] * (1-2*TEST_SIZE_RATIO)):int(texts.shape[0] * (1-TEST_SIZE_RATIO))]


    # read labels
    labels = texts_encoder(texts, first_index=FIRST_INDEX,
                                  space_index=SPACE_INDEX,
                                  space_token=SPACE_TOKEN)

    train_labels = labels[:int(labels.shape[0] * (1-2*TEST_SIZE_RATIO))]
    test_labels = labels[int(labels.shape[0] * (1-TEST_SIZE_RATIO)):] 
    validation_labels = labels[int(labels.shape[0] * (1-2*TEST_SIZE_RATIO)):int(labels.shape[0] * (1-TEST_SIZE_RATIO))] 




    NUM_FEATURES = train_inputs.shape[2] 


 

    graph = tf.Graph()
    with graph.as_default():

        print ("Building graph...")
        
        inputs_placeholder = tf.placeholder(tf.float32, [None, None, NUM_FEATURES])

        # SparseTensor placeholder required by ctc_loss op
        labels_placeholder = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size].
        sequence_length_placeholder = tf.placeholder(tf.int32, [None])
        '''
        # create a BasicRNNCell
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(NUM_HIDDEN)

        # defining initial state
        initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
 
        
        # 'inpus_placeholder' is a tensor of shape [batch_size, max_time, cell_state_size]
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        # 'state' is a tensor of shape [batch_size, cell_state_size]
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs_placeholder, initial_state=initial_state, sequence_length=sequence_length_placeholder, dtype=tf.float32)
        '''

        # Defining the cell.
        cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)

        # Stacking rnn cells.
        stack = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS,
                                            state_is_tuple=True)

        # Creates a recurrent neural network.
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs_placeholder, sequence_length_placeholder, dtype=tf.float32)


      
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

        optimizer = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, MOMENTUM).minimize(cost)

        # CTC decoder.
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, sequence_length_placeholder)

        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                           labels_placeholder))



        print ("Finished building graph.")

    with tf.Session(graph=graph) as session:

        print ("Running session ...BATCH_SIZE:", BATCH_SIZE, "NUM_EPOCHS:", NUM_EPOCHS)

        # Saver op to save and restore all the variables.
        saver = tf.train.Saver()

       # Initialize the weights and biases.     
        tf.global_variables_initializer().run()

        train_num = train_inputs.shape[0]
        num_batches_per_epoch = int(math.ceil(train_num *1.0/ BATCH_SIZE))

        train_cost_list = []
        train_label_error_rate_list = []



        for current_epoch in range(NUM_EPOCHS):
            
            train_cost = 0
            train_label_error_rate = 0
            start_time = time.time()

            for step in range(num_batches_per_epoch):
                # Format batches.
                if int(train_num / ((step + 1) * BATCH_SIZE)) >= 1:
                    indexes = [i % train_num for i in range(step * BATCH_SIZE, (step + 1) * BATCH_SIZE)]
                else:
                    indexes = [i % train_num for i in range(step * BATCH_SIZE, train_num)]

                batch_train_inputs = train_inputs[indexes]
                batch_train_sequence_lengths = train_sequence_lengths[indexes]
                batch_train_targets = sparse_tuples_from_sequences(train_labels[indexes])

                feed = {inputs_placeholder: batch_train_inputs,
                        labels_placeholder: batch_train_targets,
                        sequence_length_placeholder: batch_train_sequence_lengths}
                # print ("batch_train_inputs.shape", batch_train_inputs.shape)
                # print ("labels_placeholder.shape", labels_placeholder.get_shape())
                # print ("sequence_length_placeholder.shape", sequence_length_placeholder.shape)


                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE

                train_label_error_rate += session.run(label_error_rate, feed_dict=feed) * BATCH_SIZE


            train_cost /= train_num
            train_label_error_rate /= train_num

            # validation_feed = {inputs_placeholder: validation_inputs,
            #                    labels_placeholder: validation_labels,
            #                    sequence_length_placeholder: validation_sequence_lengths}

            # validation_cost, validation_label_error_rate = session.run([cost, label_error_rate],
            #                                                            feed_dict=validation_feed)

            # validation_cost /= validation_num
            # validation_label_error_rate /= validation_num


            if current_epoch % 100 == 0:
                D = session.run(decoded[0], feed_dict=feed)

                dense_D = tf.sparse_tensor_to_dense(D, default_value=-1).eval(session=session)

                dt = sequence_decoder(dense_D[0])
                print ("1000-------", dt, train_texts[0])

            if current_epoch % 50 == 0:
                print ("Epoch {}/{}".format(current_epoch + 1, NUM_EPOCHS))
                print ("Train cost: {}, train label error rate: {}".format(train_cost, train_label_error_rate))
                # print ("Validation cost: {}, validation label error rate: {}".format(validation_cost, validation_label_error_rate))

            # Write logs at every iteration.
            train_cost_list.append([current_epoch, train_cost])
            train_label_error_rate_list.append([current_epoch, train_label_error_rate])

            if current_epoch % 500 == 0:
                saver.save(session, MODEL_NAME+"_"+str(current_epoch))
                

        print ("Training time: ", time.time()-start_time)


        test_num = test_inputs.shape[0]
        r = BATCH_SIZE/test_num -1
        test_inputs_pl = test_inputs
        test_sequence_lengths_pl = test_sequence_lengths
        for i in range(r):
            test_inputs_pl = np.vstack((test_inputs_pl, test_inputs))
            test_sequence_lengths_pl = np.append(test_sequence_lengths_pl, test_sequence_lengths)


        num_batches_per_test = int(math.ceil(test_num *1.0/ BATCH_SIZE))

        for step in range(num_batches_per_test):
            # Format batches.
            if int(test_num / ((step + 1) * BATCH_SIZE)) >= 1:
                indexes = [i % test_num for i in range(step * BATCH_SIZE, (step + 1) * BATCH_SIZE)]
            else:
                indexes = [i % test_num for i in range(step * BATCH_SIZE, test_num)]

            batch_test_inputs = test_inputs[indexes]
            batch_test_sequence_lengths = test_sequence_lengths[indexes]
            batch_test_targets = sparse_tuples_from_sequences(test_labels[indexes])


            test_feed = {inputs_placeholder: batch_test_inputs,
                         sequence_length_placeholder: batch_test_sequence_lengths}
            # Decoding.
            decoded_outputs = session.run(decoded[0], feed_dict=test_feed)
            dense_decoded = tf.sparse_tensor_to_dense(decoded_outputs, default_value=-1).eval(session=session)

            result = []
            for i, sequence in enumerate(dense_decoded):
                if i >= test_num:
                    break
                sequence = [s for s in sequence if s != -1]
        

                decoded_text = sequence_decoder(sequence)

                seq = "Sequence {}/{}\n".format(i + 1+ step*BATCH_SIZE, test_num)
                org = "Original:\n{}\n".format(train_texts[step*BATCH_SIZE + i])
                dec = "Decoded:\n{}\n".format(decoded_text)
                print (seq)
                print (org)
                print (dec)

                result.append(seq)
                result.append(org)
                result.append(dec)
                

        # Save model weights to disk.
        save_file = saver.save(session, MODEL_NAME+"_complete")

        # write log files
        with open(RESULT_PATH + "train_cost.csv", "w") as f:
            f.write("iteration,cost\n")
            for iter, loss in train_cost_list:
                f.write("%d,%f\n" % (iter, loss))

        with open(RESULT_PATH + "train_label_error_rate.csv", "w") as f:
            f.write("iteration, error_rate\n")
            for iter, err in train_label_error_rate_list:
                f.write("%d,%f\n" % (iter, err))

        with open(RESULT_PATH + "result.txt", "w") as f:
            for r in result:
                f.write(r)

  
        print ("Model saved in file: %s", save_file)  



       


if __name__ == '__main__':
    tf.app.run()