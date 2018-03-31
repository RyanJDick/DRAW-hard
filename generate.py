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
tf.flags.DEFINE_string("write_attn", "no_attn", "Specify type of write attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset to train the model with." +
    " Options include: 'mnist', 'svhn'")

FLAGS = tf.flags.FLAGS

## GENERATIVE PARAMETERS ##
batch_size = 100

# Select dataset:
if FLAGS.dataset == 'mnist':
    dimensions = (batch_size,) + data_loader.MNISTLoader.dimensions
elif FLAGS.dataset == 'svhn':
    dimensions = (batch_size,) + data_loader.SVHNLoader.dimensions
else:
    print("dataset parameter was not recognized. Defaulting to 'mnist'.")
    dimensions = (batch_size,) + data_loader.MNISTLoader.dimensions

## CREATE MODEL ##
model = draw.DRAWGenerativeModel(FLAGS.write_attn, dimensions)

## Generate Images ##
with tf.Session() as sess:
    # Restore trained model from checkpoint
    ckpt_file = os.path.join(FLAGS.data_dir, "draw_model.ckpt")
    model.restore_from_ckpt(sess, ckpt_file)

    canvases, w_params = model.generate_images(sess)

    canvases = np.array(canvases) # T x B x H x W x C
    w_params = np.array(w_params) # T x B x num_w_params

    out_file = os.path.join(FLAGS.data_dir, "draw_generated_images.npz")
    np.savez(out_file, img=canvases, w_params=w_params)
    print("Images saved in file: %s" % out_file)
