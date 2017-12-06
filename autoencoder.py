from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import mfcc

mode = "pred" # pred or train
model_path = "autoencoder/model_3000.ckpt"
inputs = np.load("data/mfcc/inputs.npy")
test_size_ratio = 0.1
train_input = inputs[:int(inputs.shape[0] * (1-test_size_ratio))]
test_input = inputs[int(inputs.shape[0] * (1-test_size_ratio)):]

#Read input data
X = train_input.reshape(train_input.shape[0], -1)
X = np.float32(X)

num_input = X.shape[0]

#Read testing data
X_test = test_input.reshape(test_input.shape[0], -1)
X_test = np.float32(X_test)

#Read pred data
X_pred = inputs.reshape(inputs.shape[0], -1)
X_pred = np.float32(X_pred)

# Training Parameters
learning_rate = 0.01
num_steps = 3000
batch_size = 256

display_step = 10
test_step_interval = 50

# Network Parameters
input_dimension = int(X.shape[1])
#hidden_1_dimension = int(input_dimension*0.7)
#hidden_2_dimension = int(input_dimension*0.4)
hidden_1_dimension = 256
hidden_2_dimension = 128


# tf Graph input (only pictures)
encode_input = tf.placeholder("float", [None, input_dimension])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([input_dimension, hidden_1_dimension])),
    'encoder_h2': tf.Variable(tf.random_normal([hidden_1_dimension, hidden_2_dimension])),
    'decoder_h1': tf.Variable(tf.random_normal([hidden_2_dimension, hidden_1_dimension])),
    'decoder_h2': tf.Variable(tf.random_normal([hidden_1_dimension, input_dimension])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([hidden_1_dimension])),
    'encoder_b2': tf.Variable(tf.random_normal([hidden_2_dimension])),
    'decoder_b1': tf.Variable(tf.random_normal([hidden_1_dimension])),
    'decoder_b2': tf.Variable(tf.random_normal([input_dimension])),
}

# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(encode_input)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = tf.placeholder("float",[None, input_dimension])


# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
train_loss_list = []
test_loss_list = []
best_pred = []
best_loss_step = 0
best_loss = float("inf")
test_loss = 0
train_loss = 0
output_file_name = "result/auto_encoder_result.txt"

saver = tf.train.Saver()

with tf.Session() as sess:
    if mode == "pred":
        saver.restore(sess, model_path)
        X_encoder = sess.run(encoder_op, feed_dict={encode_input: X_pred})
        np.save("data/autoencoder/inputs.npy", X_encoder)
    elif mode == "train":
        # Run the initializer
        sess.run(init)

        sess_start_time = time.time()

        # Training
        for i in range(1, num_steps+1):

            # for start, end in zip(range(0, num_input, batch_size), range(batch_size, num_input, batch_size)):
            _, l = sess.run([optimizer, loss], feed_dict={encode_input: X, y_true: X})

            X_pred = sess.run(decoder_op, feed_dict={encode_input: X, y_true:X})
            train_loss = l


            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, train_loss))

            train_loss_list.append((i,train_loss))

            if i % test_step_interval == 0 or i==num_steps:
                X_pred_test = sess.run(decoder_op, feed_dict={encode_input: X_test})
                X_encoder = sess.run(encoder_op, feed_dict={encode_input: X_test})
                test_loss = sess.run(loss, feed_dict={encode_input: X_test, y_true:X_test})
                test_loss_list.append((i,test_loss))

                print('Test Loss: %f' % (test_loss))
                save_path = saver.save(sess, "autoencoder/model_%s.ckpt" % i)
                if test_loss <= best_loss:
                    best_loss = test_loss
                    best_loss_step = i
                    best_pred = X_pred_test

        with open("result/train_loss.csv", "w") as f:
            f.write("iteration,loss\n")
            for iter, loss in train_loss_list:
                f.write("%d,%f\n" % (iter, loss))
        with open("result/test_loss.csv", "w") as f:
            f.write("iteration,loss\n")
            for iter, loss in test_loss_list:
                f.write("%d,%f\n" % (iter, loss))

        with open(output_file_name, "w") as outfile:
            outfile.write("best predicted output\n")
            for line in best_pred:
                for data in line:
                    outfile.write(str(data)+" ")
                outfile.write("\n")
            outfile.write("\ntesting loss: " + str(test_loss))
            outfile.write("\ntraining loss: " + str(train_loss))
            outfile.write("\nstep to achieve best loss: "+ str(best_loss_step))
            outfile.write("\ntraining time: " + str(time.time() - sess_start_time))







