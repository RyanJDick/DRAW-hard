#!/usr/bin/env python


""""
Script to initialize and train the DRAW network.

Example Usage:
    python train.py --data_dir=/tmp/draw --read_attn=soft_attn --write_attn=soft_attn
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

## TRAINING PARAMETERS ##
batch_size = 100  # training minibatch size
train_iters = 50

## CREATE MODEL ##
model = draw.DRAWFullModel(FLAGS.read_attn, FLAGS.write_attn)

## RUN TRAINING ##

# Select dataset:
if FLAGS.dataset == 'mnist':
    data = data_loader.MNISTLoader(FLAGS.data_dir)
else:
    print("dataset parameter was not recognized. Defaulting to 'mnist'.")
    data = data_loader.MNISTLoader(FLAGS.data_dir)

Lxs = [0] * train_iters
Lzs = [0] * train_iters

with tf.Session() as sess:
# saver.restore(sess, "/tmp/draw/drawmodel.ckpt") # to restore from model, uncomment this line
    model.initialize_variables()
    for i in range(train_iters):
        xtrain = data.next_train_batch(batch_size)
        Lxs[i], Lzs[i] = model.train_batch(sess, xtrain)
        if i % 100 == 0:
            print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))
    ckpt_file = os.path.join(FLAGS.data_dir, "draw_model.ckpt")
    model.save_ckpt(sess, ckpt_file)
