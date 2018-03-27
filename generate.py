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

FLAGS = tf.flags.FLAGS

## GENERATIVE PARAMETERS ##
batch_size = 100

## CREATE MODEL ##
model = draw.DRAWGenerativeModel(FLAGS.write_attn)

## Generate Images ##
with tf.Session() as sess:
    # Restore trained model from checkpoint
    ckpt_file = os.path.join(FLAGS.data_dir, "draw_model.ckpt")
    model.restore_from_ckpt(sess, ckpt_file)

    canvases, w_cx, w_cy, w_d, w_thick = model.generate_images(sess)

    canvases = np.array(canvases)  # T x B x H x W x C
    print("Canvas size is : " + str(canvases.shape))
    w_cx = np.array(w_cx)
    w_cy = np.array(w_cy)
    w_d = np.array(w_d)
    w_thick = np.array(w_thick)

    out_file = os.path.join(FLAGS.data_dir, "draw_generated_images.npz")
    np.savez(out_file, img=canvases, w_cx=w_cx, w_cy=w_cy, w_d=w_d, w_thick=w_thick)
    print("Images saved in file: %s" % out_file)
