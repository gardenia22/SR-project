import time
import sys
import math
import numpy as np
import tensorflow as tf
from utils import *


MODEL_PATH = "./"
MODEL_NAME = MODEL_PATH + "model"

# Data directories. TODO
DATA_DIR = "./data/"



# Accounting the 0th index + space + blank label = 28 characters
NUM_CLASSES = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters.
NUM_EPOCHS = 200
NUM_HIDDEN = 50
NUM_LAYERS = 1
BATCH_SIZE = 1

# Optimizer parameters.
INITIAL_LEARNING_RATE = 1e-2
MOMENTUM = 0.9

TEST_SIZE_RATIO = 0.1

# Readme: how datas are stored:
# data/mfcc/input_file,
# data/autoencoder/input_file
# data/text_file

def main(argv):
    # arguement option: mfcc and autoencoder 
   
    if argv[1] == 'mfcc':
        # inputs shape [num_samples, max_length, num_features]
        # sequence_length shape [num_samples,]
        inputs = np.load(DATA_DIR + "mfcc/audio_inputs.npy")

        train_inputs = inputs[:int(inputs.shape[0] * (1-TEST_SIZE_RATIO))]
        train_inputs = np.float32(train_inputs) 
        train_sequence_lengths = get_sequence_length(train_inputs)

        test_inputs = inputs[int(inputs.shape[0] * (1-TEST_SIZE_RATIO)):] 
        test_inputs = np.float32(test_inputs) 
        test_sequence_lengths = get_sequence_length(test_inputs)

        # read texts 
        texts = np.load(DATA_DIR + "texts.txt")

        train_texts = texts[:int(texts.shape[0] * (1-TEST_SIZE_RATIO))]
        test_texts = texts[int(texts.shape[0] * (1-TEST_SIZE_RATIO)):]


        # read labels
        labels = texts_encoder(texts, first_index=FIRST_INDEX,
                                      space_index=SPACE_INDEX,
                                      space_token=SPACE_TOKEN)

        train_labels = labels[:int(labels.shape[0] * (1-TEST_SIZE_RATIO))]
        test_labels = labels[int(labels.shape[0] * (1-TEST_SIZE_RATIO)):] 



        NUM_FEATURES = train_inputs.shape[2] 
 
    else: 
        #TODO autoencoder inputs
    
 

    graph = tf.Graph()
    with graph.as_default():

        print ("Building graph...")
        
        inputs_placeholder = tf.placeholder(tf.float32, [None, None, NUM_FEATURES])

        # SparseTensor placeholder required by ctc_loss op
        labels_placeholder = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size].
        sequence_length_placeholder = tf.placeholder(tf.int32, [None])

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

        # CTC decoder.
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, sequence_length_placeholder)

        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                           labels_placeholder))


        print ("Finished building graph.")

    with tf.Session(graph=graph) as session:

        print ("Running session...")

        # Saver op to save and restore all the variables.
        saver = tf.train.Saver()

       # Initialize the weights and biases.     
        tf.global_variables_initializer().run()

        train_num = train_inputs.shape[0]
        num_batches_per_epoch = int(math.ceil(train_num / BATCH_SIZE))

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


                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE
                train_label_error_rate += session.run(label_error_rate, feed_dict=feed) * BATCH_SIZE


            train_cost /= train_num
            train_label_error_rate /= train_num

            print ("Epoch %d/%d (time: %.3f s)", current_epoch + 1, NUM_EPOCHS, time.time() - start_time)
            print ("Train cost: %.3f, train label error rate: %.3f", train_cost, train_label_error_rate)
            # Write logs at every iteration.
            train_cost_list.append([current_epoch, train_cost])
            train_label_error_rate_list.append([current_epoch, train_label_error_rate])
                

        print ("Training time: ", time.time()-start_time)

        test_feed = {inputs_placeholder: test_inputs,
                     sequence_length_placeholder: test_sequence_lengths}
        # Decoding.
        decoded_outputs = session.run(decoded[0], feed_dict=test_feed)
        dense_decoded = tf.sparse_tensor_to_dense(decoded_outputs, default_value=-1).eval(session=session)
        test_num = test_texts.shape[0]

        result = []
        for i, sequence in enumerate(dense_decoded):
            sequence = [s for s in sequence if s != -1]
            decoded_text = sequence_decoder(sequence)

            print ("Sequence %d/%d", i + 1, test_num)
            print ("Original:\n%s", test_texts[i])
            print ("Decoded:\n%s", decoded_text)

            result.append("Sequence %d/%d", i + 1, test_num)
            result.append("Original:\n%s", test_texts[i])
            result.apend("Decoded:\n%s", decoded_text)
            result.append("\n")

        # Save model weights to disk.
        save_file = saver.save(session, MODEL_NAME)

        # write log files
        with open("train_cost.csv", "w") as f:
            f.write("iteration,cost\n")
            for iter, loss in train_cost_list:
                f.write("%d,%f\n" % (iter, loss))

        with open("train_label_error_rate.csv", "w") as f:
            f.write("iteration, error_rate\n")
            for iter, err in train_label_error_rate_list:
                f.write("%d,%f\n" % (iter, err))

        with open("result.txt", "w") as f:
            for r in result:
                f.write(r)

  
        print ("Model saved in file: %s", save_file)  



       


if __name__ == '__main__':
    tf.app.run()