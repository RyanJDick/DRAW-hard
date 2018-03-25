#!/usr/bin/env python


""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow with 'hard'
attention via Spatial Transformer Networks (https://arxiv.org/pdf/1506.02025.pdf).

Example Usage:
    python draw.py --data_dir=/tmp/draw --read_attn=soft_attn --write_attn=soft_attn
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
from attention.no_attn import ReadNoAttn, WriteNoAttn
from attention.soft_attn import ReadSoftAttn, WriteSoftAttn
from attention.spatial_transformer_attn import ReadSpatialTransformerAttn, WriteSpatialTransformerAttn
import data_loader

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_string("read_attn", "no_attn", "Specify type of read attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("write_attn", "no_attn", "Specify type of write attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset to train the model with." +
    " Options include: 'mnist'")

FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ##

W, H, C = 28, 28, 1  # image width, height
img_size = H * W  # the canvas size
enc_size = 256  # number of hidden units / output size in LSTM
dec_size = 256
read_n = 5  # read glimpse grid width/height
write_n = 5  # write glimpse grid width/height
z_size = 10  # QSampler output size
T = 10  # MNIST generation sequence length
batch_size = 100  # training minibatch size
train_iters = 100
learning_rate = 1e-3  # learning rate for optimizer
eps = 1e-8  # epsilon for numerical stability

## BUILD MODEL ##

x = tf.placeholder(tf.float32, shape=(batch_size, H, W, C))  # input
e = tf.random_normal((batch_size, z_size), mean=0, stddev=1)  # Qsampler noise
lstm_enc = tf.contrib.rnn.LSTMCell(enc_size, state_is_tuple=True)  # encoder Op
lstm_dec = tf.contrib.rnn.LSTMCell(dec_size, state_is_tuple=True)  # decoder Op

## READER ##
if FLAGS.read_attn == 'no_attn':
    reader = ReadNoAttn(H, W, C, read_n)
elif FLAGS.read_attn == 'soft_attn':
    reader = ReadSoftAttn(H, W, C, read_n)
elif FLAGS.read_attn == 'spatial_transformer_attn':
    read_n = 10
    reader = ReadSpatialTransformerAttn(H, W, C, read_n)
else:
    print("read_attn parameter was not recognized. Defaulting to 'no_attn'.")
    reader = ReadNoAttn(H, W, C, read_n)


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

if FLAGS.write_attn == 'no_attn':
    writer = WriteNoAttn(H, W, C, write_n)
elif FLAGS.write_attn == 'soft_attn':
    writer = WriteSoftAttn(H, W, C, write_n)
elif FLAGS.write_attn == 'spatial_transformer_attn':
    write_n = 10
    writer = WriteSpatialTransformerAttn(H, W, C, write_n)
else:
    print("write_attn parameter was not recognized. Defaulting to 'no_attn'.")
    writer = WriteNoAttn(H, W, C, write_n)


## STATE VARIABLES ##

cs = [0] * T  # sequence of canvases
# gaussian params generated by SampleQ. We will need these for computing loss.
mus, logsigmas, sigmas = [0] * T, [0] * T, [0] * T
# attention window visualization parameters
r_cxs, r_cys, r_ds, r_thickness = [0] * T, [0] * T, [0] * T, [0] * T
w_cxs, w_cys, w_ds, w_thickness = [0] * T, [0] * T, [0] * T, [0] * T

# initial states
h_dec_prev = tf.zeros((batch_size, dec_size))
enc_state = lstm_enc.zero_state(batch_size, tf.float32)
dec_state = lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ##

# Construct the unrolled computational graph
with tf.variable_scope("draw", reuse=tf.AUTO_REUSE):
    for t in range(T):
        c_prev = tf.zeros((batch_size, H, W, C)) if t == 0 else cs[t - 1]
        x_hat = x - tf.sigmoid(c_prev)  # error image
        r, r_cxs[t], r_cys[t], r_ds[t], r_thickness[t] = reader.read(x, x_hat, h_dec_prev)
        h_enc, enc_state = encode(enc_state, tf.concat([r, h_dec_prev], 1))
        z, mus[t], logsigmas[t], sigmas[t] = sampleQ(h_enc)
        h_dec, dec_state = decode(dec_state, z)
        w, w_cxs[t], w_cys[t], w_ds[t], w_thickness[t] = writer.write(h_dec)
        cs[t] = c_prev + w
        h_dec_prev = h_dec


## LOSS FUNCTION ##

def binary_crossentropy(t, o):
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


# reconstruction term appears to have been collapsed down to a single scalar
# value (rather than one per item in minibatch)
x_recons = tf.nn.sigmoid(cs[-1])

# After computing binary cross entropy, sum across features then take the mean
# of those sums across minibatches
Lx = tf.reduce_sum(binary_crossentropy(x, x_recons), [1, 2])  # reconstruction term
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

# Select dataset:
if FLAGS.dataset == 'mnist':
    data = data_loader.MNISTLoader(FLAGS.data_dir)
else:
    print("dataset parameter was not recognized. Defaulting to 'mnist'.")
    data = data_loader.MNISTLoader(FLAGS.data_dir)

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
    xtrain = data.next_train_batch(batch_size)
    feed_dict = {x: xtrain}
    results = sess.run(fetches, feed_dict)
    Lxs[i], Lzs[i], _ = results
    if i % 100 == 0:
        print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))

## TRAINING FINISHED ##
# Generate some examples
canvases, r_cx, r_cy, r_d, r_thick, w_cx, w_cy, w_d, w_thick = sess.run(
    [cs, r_cxs, r_cys, r_ds, r_thickness, w_cxs, w_cys, w_ds, w_thickness], feed_dict)
canvases = np.array(canvases)  # T x B x H x W x C
r_cx = np.array(r_cx)
r_cy = np.array(r_cy)
r_d = np.array(r_d)
r_thick = np.array(r_thick)
w_cx = np.array(w_cx)
w_cy = np.array(w_cy)
w_d = np.array(w_d)
w_thick = np.array(w_thick)

out_file = os.path.join(FLAGS.data_dir, "draw_data.npy")
np.save(out_file, [canvases, r_cx, r_cy, r_d, r_thick, w_cx, w_cy, w_d, w_thick, Lxs, Lzs])
print("Outputs saved in file: %s" % out_file)

ckpt_file = os.path.join(FLAGS.data_dir, "draw_model.ckpt")
print("Model saved in file: %s" % saver.save(sess, ckpt_file))

sess.close()
