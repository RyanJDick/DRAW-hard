#!/usr/bin/env python


""""
Script to initialize and train the DRAW network.

Example Usage:
    python train.py --data_dir=/tmp/draw --model_dir=./out --dataset=svhn --read_attn=soft_attn --write_attn=soft_attn
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
import shutil

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_string("model_dir", "", "")
tf.flags.DEFINE_string("read_attn", "no_attn", "Specify type of read attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("write_attn", "no_attn", "Specify type of write attention " +
    "to use. Options include: 'no_attn', 'soft_attn', 'spatial_transformer_attn'.")
tf.flags.DEFINE_string("dataset", "mnist", "Dataset to train the model with." +
    " Options include: 'mnist', 'svhn'")

FLAGS = tf.flags.FLAGS

## TRAINING PARAMETERS ##
batch_size = 100  # training minibatch size
train_iters = 10000

## RUN TRAINING ##

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

# Create model directory if it does not exist:
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
# Copy draw.py to the model directory so there is a record of all hyperparameters used:
draw_copy_file = os.path.join(FLAGS.model_dir, "draw_model_copy.txt")
shutil.copy2('draw.py', draw_copy_file)

Lxs = [0] * train_iters
Lzs = [0] * train_iters

ckpt_file = os.path.join(FLAGS.model_dir, "draw_model.ckpt")

with tf.Session() as sess:
# saver.restore(sess, "/tmp/draw/drawmodel.ckpt") # to restore from model, uncomment this line
    model.initialize_variables()
    for i in range(train_iters):
        xtrain = data.next_train_batch(batch_size)
        Lxs[i], Lzs[i] = model.train_batch(sess, xtrain)
        if i % 100 == 0:
            print("iter=%d : Lx: %f Lz: %f" % (i, Lxs[i], Lzs[i]))
        if i % 1000 == 0: # save checkpoint every 1000 iterations
            model.save_ckpt(sess, ckpt_file)

    model.save_ckpt(sess, ckpt_file)

    Lxs = np.array(Lxs)
    Lzs = np.array(Lzs)

    out_file = os.path.join(FLAGS.model_dir, "train_loss.npz")
    np.savez(out_file, Lxs=Lxs, Lzs=Lzs)
    print("Training loss saved in file: %s" % out_file)
