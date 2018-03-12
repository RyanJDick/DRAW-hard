#!/usr/bin/env python


""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow with 'hard'
attention via Spatial Transformer Networks (https://arxiv.org/pdf/1506.02025.pdf).

Example Usage:
    python draw.py --data_dir=data --write_same_attn=False
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
from transformer import transformer

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("write_same_attn", False, "Whether write step should be forced to write to the same attention patch as the read step.")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ##

W, H = 28, 28  # image width,height
img_size = H * W  # the canvas size
enc_size = 256  # number of hidden units / output size in LSTM
dec_size = 256
read_n = 5  # read glimpse grid width/height
write_n = 5  # write glimpse grid width/height
read_size = 2 * read_n * read_n
write_size = write_n * write_n
z_size = 10  # QSampler output size
T = 10  # MNIST generation sequence length
batch_size = 100  # training minibatch size
train_iters = 10000
learning_rate = 1e-2  # learning rate for optimizer
eps = 1e-8  # epsilon for numerical stability

## BUILD MODEL ##

x = tf.placeholder(tf.float32, shape=(batch_size, img_size))  # input (batch_size * img_size)
e = tf.random_normal((batch_size, z_size), mean=0, stddev=1)  # Qsampler noise
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)  # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)  # decoder Op


def spatial_transformer_attn_window_params(scope, h_dec):
    '''
    Apply a localization network with a single hidden layer to the h_dec output
    in order to determine the parameters used to define the attention window.
    The parameters to be predicted are:
    s - scaling parameter
    tx - x translation
    ty - y translation

    These parameters are used to define an affine transformation to be applied
    by a spatial transformer network as follows:
    A = [[s, 0, tx],
         [0, s, ty]]

    The fully-connected layer is initialized to achieved an identity affine
    transformation (no transformation) before training:
    A0 = [[1, 0, 0],
          [0, 1, 0]]
    '''
    # Define 'localization network':
    with tf.variable_scope(scope):
        with tf.variable_scope('scale'):
            scale = tf.contrib.layers.fully_connected(h_dec, 1,
                activation_fn=tf.nn.sigmoid, # Limit scale to (0,1)
                weights_initializer=tf.zeros_initializer,
                biases_initializer=tf.ones_initializer, # Initially, output scale = 1
                scope='fc')
            s = scale[:, 0]
        with tf.variable_scope('shift'):
            shift = tf.contrib.layers.fully_connected(h_dec, 2,
                activation_fn=tf.nn.tanh, # Limit translation to (-1, 1)
                weights_initializer=tf.zeros_initializer,
                biases_initializer=tf.zeros_initializer, # Initially, output shift = 0
                scope='fc')
            tx, ty = shift[:, 0], shift[:, 1]
        return s, tx, ty


def params_to_transformation_matrix(s, tx, ty):
    """
    Combine scale and translation into transformation matrix

    This operation produces the following tensor (for a batch size of 2):
    [[[s1,  0, tx1],  # batch 1
      [ 0, s1, ty1]],
     [[s2,  0, tx2],  # batch 2
      [ 0, s2, ty2]]]
    """
    return tf.stack([
        tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(tx, 1)], axis=1),
        tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(ty, 1)], axis=1),
        ], axis=1)

## READ ##

def read_spatial_transformer_attention(x, x_hat, s, tx, ty):
    s, tx, ty = spatial_transformer_attn_window_params('read_attn', h_dec_prev)
    transformation_mat = params_to_transformation_matrix(s, tx, ty)
    with tf.variable_scope('read_spatial_transformer'):
        # Full canvas to attention window transformation
        x = tf.expand_dims(tf.reshape(x, [batch_size, H, W]), -1)
        x_hat = tf.expand_dims(tf.reshape(x_hat, [batch_size, H, W]), -1)
        x_window = transformer(x, transformation_mat, [read_n, read_n])
        x_window = tf.reshape(x_window, [batch_size, read_n * read_n])
        x_hat_window = transformer(x_hat, transformation_mat, [read_n, read_n])
        x_hat_window = tf.reshape(x_hat_window, [batch_size, read_n * read_n])
        return tf.concat([x_window, x_hat_window], 1)


## ENCODE ##

def encode(state, input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder"):
        return lstm_enc(input, state)


## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("Qsampler"):
        with tf.variable_scope("mu"):
            mu = tf.contrib.layers.fully_connected(h_enc, z_size, activation_fn=None)
        with tf.variable_scope("sigma"):
            logsigma = tf.contrib.layers.fully_connected(h_enc, z_size, activation_fn=None)
            sigma = tf.exp(logsigma)
    return (mu + sigma * e, mu, logsigma, sigma)

## DECODER ##

def decode(state, input):
    with tf.variable_scope("decoder"):
        return lstm_dec(input, state)

## WRITER ##

def write_spatial_transformer_attention(h_dec, s, tx, ty):
    with tf.variable_scope("write"):
        w = tf.contrib.layers.fully_connected(h_dec, write_size, activation_fn=None, scope='fc_write_contents') # batch x (write_n*write_n)
        w = tf.reshape(w, [batch_size, write_n, write_n])
        transformation_mat = params_to_transformation_matrix(s, tx, ty)

        # Attention window to full canvas size
        canvas = transformer(tf.expand_dims(w, -1), transformation_mat, [H, W])
        canvas = tf.reshape(canvas, [batch_size, H * W])
        return canvas

## STATE VARIABLES ##

cs = [0] * T  # sequence of canvases
# gaussian params generated by SampleQ. We will need these for computing loss.
mus, logsigmas, sigmas = [0] * T, [0] * T, [0] * T
# attention window transformation PARAMETERS
read_scales, read_txs, read_tys = [1] * T, [0] * T, [0] * T
write_scales, write_txs, write_tys = [1] * T, [0] * T, [0] * T
# initial states
h_dec_prev = tf.zeros((batch_size, dec_size))
enc_state = lstm_enc.zero_state(batch_size, tf.float32)
dec_state = lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ##

# Construct the unrolled computational graph
with tf.variable_scope("draw", reuse=tf.AUTO_REUSE):
    for t in range(T):
        c_prev = tf.zeros((batch_size, img_size)) if t == 0 else cs[t - 1]
        x_hat = x - tf.sigmoid(c_prev)  # error image
        read_scales[t], read_txs[t], read_tys[t] = spatial_transformer_attn_window_params("read_attn", h_dec_prev)
        r = read_spatial_transformer_attention(x, x_hat, read_scales[t], read_txs[t], read_tys[t])
        h_enc, enc_state = encode(enc_state, tf.concat([r, h_dec_prev], 1))
        z, mus[t], logsigmas[t], sigmas[t] = sampleQ(h_enc)
        h_dec, dec_state = decode(dec_state, z)
        if FLAGS.write_same_attn:
            write_scales[t] = 1.0 / read_scales[t]
            write_txs[t] = -read_txs[t] / read_scales[t]
            write_tys[t] = -read_tys[t] / read_scales[t]
        else:
            write_scales[t], write_txs[t], write_tys[t] = spatial_transformer_attn_window_params("write_attn", h_dec)
        cs[t] = c_prev + write_spatial_transformer_attention(h_dec, write_scales[t], write_txs[t], write_tys[t])  # store results
        h_dec_prev = h_dec

## LOSS FUNCTION ##

def binary_crossentropy(t, o):
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons = tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx = tf.reduce_sum(binary_crossentropy(x, x_recons), 1)  # reconstruction term
Lx = tf.reduce_mean(Lx)

kl_terms = [0] * T
for t in range(T):
    mu2 = tf.square(mus[t])
    sigma2 = tf.square(sigmas[t])
    logsigma = logsigmas[t]
    # each kl term is (1xminibatch)
    kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - .5
# this is 1xminibatch, corresponding to summing kl_terms from 1:T
KL = tf.add_n(kl_terms)
Lz = tf.reduce_mean(KL)  # average over minibatches

cost = Lx + Lz

## OPTIMIZER ##

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
grads = optimizer.compute_gradients(cost)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
train_op = optimizer.apply_gradients(grads)

## RUN TRAINING ##

data_directory = os.path.join(FLAGS.data_dir, "mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(
    data_directory, one_hot=True).train  # binarized (0-1) mnist data

fetches = []
fetches.extend([Lx, Lz, train_op])
Lxs = [0] * train_iters
Lzs = [0] * train_iters

sess = tf.InteractiveSession()

saver = tf.train.Saver()  # saves variables learned during training
tf.global_variables_initializer().run()
# saver.restore(sess, "/tmp/draw/drawmodel.ckpt") # to restore from model, uncomment this line

for i in range(train_iters):
    # xtrain is (batch_size x img_size)
    xtrain, _ = train_data.next_batch(batch_size)
    feed_dict = {x: xtrain}
    results = sess.run(fetches, feed_dict)
    Lxs[i], Lzs[i], _ = results
    if i % 100 == 0:
        print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))

## TRAINING FINISHED ##

canvases = sess.run(cs, feed_dict)  # generate some examples
canvases = np.array(canvases)  # T x batch x img_size

out_file = os.path.join(FLAGS.data_dir, "draw_transformer_data.npy")
np.save(out_file, [canvases, Lxs, Lzs])
print("Outputs saved in file: %s" % out_file)

ckpt_file = os.path.join(FLAGS.data_dir, "draw_transformer_model.ckpt")
print("Model saved in file: %s" % saver.save(sess, ckpt_file))

sess.close()
