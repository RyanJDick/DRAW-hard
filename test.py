#!/usr/bin/env python


""""
Script to evaluate a trained DRAW network.

Example Usage:
    python test.py --data_dir=/tmp/draw --read_attn=soft_attn --write_attn=soft_attn
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
tf.flags.DEFINE_string("read_attn", "no_attn", "Specify type of read attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("write_attn", "no_attn", "Specify type of write attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset to train the model with." +
    " Options include: 'mnist'")

FLAGS = tf.flags.FLAGS

## TESTING PARAMETERS ##
batch_size = 100  # training minibatch size

## CREATE MODEL ##
model = draw.DRAWFullModel(FLAGS.read_attn, FLAGS.write_attn)

## TEST TRAINED MODEL ##

# Select dataset:
if FLAGS.dataset == 'mnist':
    data = data_loader.MNISTLoader(FLAGS.data_dir)
else:
    print("dataset parameter was not recognized. Defaulting to 'mnist'.")
    data = data_loader.MNISTLoader(FLAGS.data_dir)

sess = tf.InteractiveSession()
with tf.Session() as sess:
    # Restore trained model from checkpoint
    ckpt_file = os.path.join(FLAGS.data_dir, "draw_model.ckpt")
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
    canvases, r_cx, r_cy, r_d, r_thick, w_cx, w_cy, w_d, w_thick = model.generate_examples(sess, xtest)

    canvases = np.array(canvases)  # T x B x H x W x C
    r_cx = np.array(r_cx)
    r_cy = np.array(r_cy)
    r_d = np.array(r_d)
    r_thick = np.array(r_thick)
    w_cx = np.array(w_cx)
    w_cy = np.array(w_cy)
    w_d = np.array(w_d)
    w_thick = np.array(w_thick)


    out_file = os.path.join(FLAGS.data_dir, "draw_examples.npz")
    np.savez(out_file, img=canvases, r_cx=r_cx, r_cy=r_cy, r_d=r_d, r_thick=r_thick, w_cx=w_cx, w_cy=w_cy, w_d=w_d, w_thick=w_thick)
    print("Outputs saved in file: %s" % out_file)
