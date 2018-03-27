#!/usr/bin/env python


""""
Simple implementation of http://arxiv.org/pdf/1502.04623v2.pdf in TensorFlow with 'hard'
attention via Spatial Transformer Networks (https://arxiv.org/pdf/1506.02025.pdf).
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
from attention.no_attn import ReadNoAttn, WriteNoAttn
from attention.soft_attn import ReadSoftAttn, WriteSoftAttn
from attention.spatial_transformer_attn import ReadSpatialTransformerAttn, WriteSpatialTransformerAttn
import data_loader

class DRAW:

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
    learning_rate = 1e-3  # learning rate for optimizer
    eps = 1e-8  # epsilon for numerical stability
    read_attn = ""
    write_attn = ""


    def __init__(self, read_attn, write_attn):
        self.read_attn = read_attn
        self.write_attn = write_attn


    def _sampleQ(self, h_enc, e):
        """
        Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
        mu is (batch,z_size)
        """
        with tf.variable_scope("Qsampler"):
            with tf.variable_scope("mu"):
                mu = tf.contrib.layers.fully_connected(h_enc, self.z_size, activation_fn=None)
            with tf.variable_scope("sigma"):
                logsigma = tf.contrib.layers.fully_connected(h_enc, self.z_size, activation_fn=None)
                sigma = tf.exp(logsigma)
        return (mu + sigma * e, mu, logsigma, sigma)


    def _loss(self, x, x_recons, mus, sigmas, logsigmas):
        def binary_crossentropy(t, o):
            return -(t * tf.log(o + self.eps) + (1.0 - t) * tf.log(1.0 - o + self.eps))

        # reconstruction term appears to have been collapsed down to a single scalar
        # value (rather than one per item in minibatch)
        x_recons = tf.nn.sigmoid(x_recons)

        # After computing binary cross entropy, sum across features then take the mean
        # of those sums across minibatches
        Lx = tf.reduce_sum(binary_crossentropy(x, x_recons), [1, 2])  # reconstruction term
        Lx = tf.reduce_mean(Lx)

        kl_terms = [0] * self.T
        for t in range(self.T):
            mu2 = tf.square(mus[t])
            sigma2 = tf.square(sigmas[t])
            logsigma = logsigmas[t]
            # each kl term is (1xminibatch)
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - .5
        # this is 1xminibatch, corresponding to summing kl_terms from 1:T
        KL = tf.add_n(kl_terms)
        Lz = tf.reduce_mean(KL)  # average over minibatches

        return Lx, Lz


    def _optimizer(self, cost):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        grads = optimizer.compute_gradients(cost)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads)
        return train_op


    def _get_reader(self):
        """
        Return the appropriate reader based on the specified attention type.
        """
        if self.read_attn == 'no_attn':
            reader = ReadNoAttn(self.H, self.W, self.C, self.read_n)
        elif self.read_attn == 'soft_attn':
            reader = ReadSoftAttn(self.H, self.W, self.C, self.read_n)
        elif self.read_attn == 'spatial_transformer_attn':
            self.read_n = 10
            reader = ReadSpatialTransformerAttn(self.H, self.W, self.C, self.read_n)
        else:
            print("read_attn parameter was not recognized. Defaulting to 'no_attn'.")
            reader = ReadNoAttn(self.H, self.W, self.C, self.read_n)
        return reader

    def _get_writer(self):
        """
        Return the appropriate writer based on the specified attention type.
        """
        if self.write_attn == 'no_attn':
            writer = WriteNoAttn(self.H, self.W, self.C, self.write_n)
        elif self.write_attn == 'soft_attn':
            writer = WriteSoftAttn(self.H, self.W, self.C, self.write_n)
        elif self.write_attn == 'spatial_transformer_attn':
            write_n = 10
            writer = WriteSpatialTransformerAttn(self.H, self.W, self.C, self.write_n)
        else:
            print("write_attn parameter was not recognized. Defaulting to 'no_attn'.")
            writer = WriteNoAttn(self.H, self.W, self.C, self.write_n)
        return writer

    def _encoder_model(self, t, e, c_prev, lstm_enc, enc_state, h_dec_prev, reader):
        """
        Construct the encoder portion of the iterative DRAW model.

        Parameters:
        -----------

        t :             Current time step

        e :             Gaussian noise for Q sampler

        c_prev:         Canvas after previous time step.

        lstm_enc:       Encoder LSTM cell.

        enc_state:      LSTM encoder state from previous time step.

        h_dec_prev:     Output from LSTM decoder in the previous time step.

        reader:         The attentive writer to use.

        Return:
        ------
        (z, enc_state)
        z:              Sampled latent representation.

        enc_state:      State of LSTM encoder.
        """
        with tf.variable_scope("encoder"):
            x_hat = self.x - tf.sigmoid(c_prev)  # error image
            r, self.read_params[t] = reader.read(self.x, x_hat, h_dec_prev)
            with tf.variable_scope("encoder_lstm"):
                h_enc, enc_state = lstm_enc(tf.concat([r, h_dec_prev], 1), enc_state)
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self._sampleQ(h_enc, e)
            return (z, enc_state)

    def _decoder_model(self, t, z, lstm_dec, dec_state, writer):
        """
        Construct the decoder portion of the iterative DRAW model.

        Parameters:
        -----------
        t:              Current time step.

        z :             Sampled latent variable.

        lstm_dec:       The LSTM cell.

        dec_state:      LSTM decoder state from previous time step.

        writer:         The attentive writer to use.

        Return:
        -------
        (w, h_dec, dec_state)

        w:              Update to write to the canvas.

        h_dec:          Output from LSTM decoder. Used as input to encoder in
                        next time step.

        dec_state:      State of LSTM decoder.
        """
        with tf.variable_scope("decoder"):
            with tf.variable_scope("decoder_lstm"):
                h_dec, dec_state = lstm_dec(z, dec_state)
            w, self.write_params[t] = writer.write(h_dec)
            return (w, h_dec, dec_state)

    def restore_from_ckpt(self, sess, ckpt_file):
        self.saver.restore(sess, ckpt_file)
        print("Model restored from: " + ckpt_file)

    def initialize_variables(self):
        tf.global_variables_initializer().run()

### DRAW Full Model (Encoder and Decoder) ###
class DRAWFullModel(DRAW):
    def __init__(self, read_attn, write_attn):
        super(DRAWFullModel, self).__init__(read_attn, write_attn)
        self._draw_full_model() # Construct the unrolled computation graph

    def _draw_full_model(self):
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.H, self.W, self.C))  # input
        e = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)  # Qsampler noise
        lstm_enc = tf.contrib.rnn.LSTMCell(self.enc_size, state_is_tuple=True)  # encoder Op
        lstm_dec = tf.contrib.rnn.LSTMCell(self.dec_size, state_is_tuple=True)  # decoder Op

        reader = self._get_reader()
        writer = self._get_writer()

        self.cs = [0] * self.T  # Sequence of canvases

        # Gaussian params generated by SampleQ. We will need these for computing loss.
        self.mus, self.logsigmas, self.sigmas = [0] * self.T, [0] * self.T, [0] * self.T

        # Attention window visualization parameters
        self.read_params = [0] * self.T
        self.write_params = [0] * self.T

        # initial states
        h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
        enc_state = lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = lstm_dec.zero_state(self.batch_size, tf.float32)

        # Construct the unrolled computational graph
        with tf.variable_scope("draw", reuse=tf.AUTO_REUSE):
            for t in range(self.T):
                c_prev = tf.zeros((self.batch_size, self.H, self.W, self.C)) if t == 0 else self.cs[t - 1]
                z, enc_state = self._encoder_model(t, e, c_prev, lstm_enc, enc_state, h_dec_prev, reader)
                w, h_dec, dec_state = self._decoder_model(t, z, lstm_dec, dec_state, writer)
                self.cs[t] = c_prev + w
                h_dec_prev = h_dec

            self.Lx, self.Lz = self._loss(self.x, self.cs[-1], self.mus, self.sigmas, self.logsigmas)
            cost = self.Lx + self.Lz
        with tf.variable_scope("optimizer"):
            self.train_op = self._optimizer(cost)

        # Full saver:
        self.saver = tf.train.Saver()

    def train_batch(self, sess, batch):
        """
        Run the training operation on the provided batch of training data, and
        report the training loss.

        Parameters:
        -----------
        sess:       Tensorflow session.

        batch:      A batch of training data of shape (batch_size, H, W, C)

        Return:
        -------
        (Lx, Lz)
        """
        feed_dict = {self.x: batch}
        recon_loss, kl_loss, _ = sess.run([self.Lx, self.Lz, self.train_op], feed_dict)
        return recon_loss, kl_loss

    def test_reconstruction_batch(self, sess, batch):
        """
        Run the training operation on the provided batch of test data, and
        report the reconstruction loss (NLL).

        Parameters:
        -----------
        sess:       Tensorflow session.

        batch:      A batch of testing data of shape (batch_size, H, W, C)

        Return:
        -------
        Lx_mean:    Mean reconstruction loss over the entire batch.
        """
        feed_dict = {self.x: batch}
        recon_loss = sess.run(self.Lx, feed_dict)
        return recon_loss

    def generate_examples(self, sess, batch):
        """
        Generate examples for visualization. This involves storing the attention
        parameters and intermediate canvases from each time step.

        Parameters:
        -----------
        sess:       Tensorflow session.

        batch:      A batch to generate examples from.
                    Shape: (batch_size, H, W, C)

        Return:
        -------
        (cs, read_params, write_params)
        """
        feed_dict = {self.x: batch}
        result = sess.run([self.cs, self.read_params, self.write_params], feed_dict)
        return result

    def save_ckpt(self, sess, ckpt_file):
        filename = self.saver.save(sess, ckpt_file)
        print("Model saved in file: " + filename)



### DRAW Generative Model (Decoder Only) ###
class DRAWGenerativeModel(DRAW):
    def __init__(self, write_attn):
        self.write_attn = write_attn
        # Construct the unrolled computation graph for the decoder portion of
        # the network only:
        self._draw_decoder_model()

    def _draw_decoder_model(self):
        # Sample latent z from normal prior with mean=0, stddev=1
        z = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)

        lstm_dec = tf.contrib.rnn.LSTMCell(self.dec_size, state_is_tuple=True)
        writer = self._get_writer()

        self.cs = [0] * self.T  # Sequence of canvases

        # Attention window visualization parameters
        self.write_params = [0] * self.T

        # Initial state
        dec_state = lstm_dec.zero_state(self.batch_size, tf.float32)

        # Construct the unrolled computational graph
        with tf.variable_scope("draw", reuse=tf.AUTO_REUSE):
            for t in range(self.T):
                c_prev = tf.zeros((self.batch_size, self.H, self.W, self.C)) if t == 0 else self.cs[t - 1]
                w, h_dec, dec_state = self._decoder_model(t, z, lstm_dec, dec_state, writer)
                self.cs[t] = c_prev + w

        # This saver will only contain the variables under the scope
        # "draw\decoder" in the full model. It can be used to restore from the
        # full checkpoint and will simply ignore the stored weights that are not
        # present in this model.
        self.saver = tf.train.Saver()


    def generate_images(self, sess):
        return sess.run([self.cs, self.write_params])
