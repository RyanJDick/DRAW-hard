#!/usr/bin/env python


""""
Script to evaluate a trained DRAW network.

Example Usage:
    python test.py --data_dir=/tmp/draw --model_dir=./out --dataset=mnist --read_attn=soft_attn --write_attn=soft_attn
"""

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os
from attention.no_attn import ReadNoAttn, WriteNoAttn
from attention.soft_attn import ReadSoftAttn, WriteSoftAttn
from attention.spatial_transformer_attn import ReadSpatialTransformerAttn, WriteSpatialTransformerAttn
import data_loader
import draw

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_string("model_dir", "", "")
tf.flags.DEFINE_string("read_attn", "no_attn", "Specify type of read attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("write_attn", "no_attn", "Specify type of write attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset to train the model with." +
    " Options include: 'mnist', 'svhn'")

FLAGS = tf.flags.FLAGS

## TESTING PARAMETERS ##
batch_size = 100  # training minibatch size

## TEST TRAINED MODEL ##

# Select dataset:
if FLAGS.dataset == 'mnist':
    data = data_loader.MNISTLoader(FLAGS.data_dir)
elif FLAGS.dataset == 'svhn':
    data = data_loader.SVHNLoader(FLAGS.data_dir)
else:
    print("dataset parameter was not recognized. Defaulting to 'mnist'.")
    data = data_loader.MNISTLoader(FLAGS.data_dir)

## CREATE MODEL ##
dimensions = (batch_size,) + data.dimensions
model = draw.DRAWFullModel(FLAGS.read_attn, FLAGS.write_attn, dimensions)

with tf.Session() as sess:
    # Restore trained model from checkpoint
    ckpt_file = os.path.join(FLAGS.model_dir, "draw_model.ckpt")
    model.restore_from_ckpt(sess, ckpt_file)

    xtest = data.next_test_batch(batch_size)
    batch = 0
    test_nll = 0
    print("Testing trained model:")
    while xtest is not None:
        batch += 1
        test_nll += model.test_reconstruction_batch(sess, xtest)
        print("Test samples: " + str(batch * batch_size) + ", Mean NLL: " + str(test_nll / batch))
        xtest = data.next_test_batch(batch_size)


    ## GENERATE EXAMPLES ##
    # Generate examples to be used for visualizing reconstruction process and the
    # attention behaviour.

    xtest = data.next_test_batch(batch_size)
    canvases, r_params, w_params = model.generate_examples(sess, xtest)

    canvases = np.array(canvases)  # T x B x H x W x C
    r_params = np.array(r_params)
    w_params = np.array(w_params)

    out_file = os.path.join(FLAGS.model_dir, "draw_examples.npz")
    np.savez(out_file, img=canvases, r_params=r_params, w_params=w_params)
    print("Outputs saved in file: %s" % out_file)
