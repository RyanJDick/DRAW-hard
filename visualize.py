"""
Takes data saved by DRAW model and generates visualizations

Example usage:
1. To create T images (at T time steps), each containing 10 x 10 = 100 sample
reconstruction images:
	python visualize.py 10imgsx100samples /tmp/draw/visualize /tmp/draw/draw_examples.npz

2. To create a single image containing all T time steps for a single
reconstruction image:
	python visualize.py 1imgx1sample /tmp/draw/visualize /tmp/draw/draw_examples.npz

3. To create a single image showing all T time steps for 10 reconstruction
images:
	python visualize.py 1imgx10samples /tmp/draw/visualize /tmp/draw/draw_examples.npz

4. To plot the training curves for multiple training runs:
	python visualize.py train_curve /tmp/draw/visualize /tmp/draw1/train_loss.npz /tmp/draw2/train_loss.npz
"""

import argparse
import matplotlib
import sys
import os
import numpy as np
from scipy.misc import imsave

matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

read_attn = True
write_attn = True

def draw_attention_box(img, attn_params, colour):
	"""
	Draws a box on the input image to illustrate the attention region.

	Parameters:
	-----------
	img :			Input image. (H, W, C)

	attn_params:	Tuple containing parameters needed to draw attention box.
					Tuple contains: (cx, cy, d, thickness):
					cx, cy:		Center coordinates of the rectangle.
					d:			Dimension of the square to draw in pixels.
					thickness:  Thickness of the rectangle to draw in pixels.

	colour:			Colour to use. (R, G, B)
	"""
	H, W, C = img.shape
	cx, cy, d, thickness = attn_params
	# Calculate boundary pixel values of all edges of the square:
	# 'out' pixel values are inclusive
	# 'in' pixel values are exclusive
	thickness = max(int(thickness), 1)
	y_top_out = np.clip(int(cy) - int(d / 2) - thickness, 0, H - 1)
	y_top_in = np.clip(y_top_out + thickness, 0, H - 1)
	y_bottom_out = np.clip(int(cy) + int(d / 2) + thickness, 0, H - 1)
	y_bottom_in = np.clip(y_bottom_out - thickness, 0, H - 1)

	x_left_out = np.clip(int(cx) - int(d / 2) - thickness, 0, W - 1)
	x_left_in = np.clip(x_left_out + thickness, 0, W - 1)
	x_right_out = np.clip(int(cx) + int(d / 2) + thickness, 0, W - 1)
	x_right_in = np.clip(x_right_out - thickness, 0, W - 1)

	# Draw top line:
	for x in range(x_left_out, x_right_out + 1):
		for y in range(y_top_out, y_top_in):
			img[x, y, :] = colour

	# Draw bottom line:
	for x in range(x_left_out, x_right_out + 1):
		for y in range(y_bottom_in + 1, y_bottom_out + 1):
			img[x, y, :] = colour

	# Draw left line:
	for x in range(x_left_out, x_left_in):
		for y in range(y_top_out, y_bottom_out + 1):
			img[x, y, :] = colour

	# Draw right line:
	for x in range(x_right_in + 1, x_right_out + 1):
		for y in range(y_top_out, y_bottom_out + 1):
			img[x, y, :] = colour

def xrecons_grid(X, read_attn_params, write_attn_params, args):
	"""
	Plots grid of canvases for a single time step.

	Parameters:
	-----------
	X :					X is x_recons from a single time step. (batch_size, H, W, C)
						Note that the batch_size is assumed to be a square number.

	read_attn_params:	Tuple containing parameters needed to draw read
						attention box. Tuple contains: (r_cx, r_cy, r_d, r_thick)

	write_attn_params:	Tuple containing parameters needed to draw read
						attention box. Tuple contains: (r_cx, r_cy, r_d, r_thick)

	args:				Command line args.
	"""
	batch_size, H, W, C = X.shape

	padsize = 1
	padval = 1.0
	ph = H + 2 * padsize  # Padded height
	pw = W + 2 * padsize  # Padded width

	# Take square root, because the image is going to contain a square grid of
	# reconstruction examples:
	N = int(np.sqrt(batch_size))
	X = X.reshape((N, N, H, W, C))
	img = np.ones((N * ph, N * pw, C)) * padval
	for i in range(N):
		for j in range(N):
			startr = i * ph + padsize
			endr = startr + H
			startc = j * pw + padsize
			endc = startc + W
			k = i * N + j # Batch index
			if args.read_attn:
				draw_attention_box(X[i, j, :, :, :], read_attn_params[k, :], np.array([1.0, 0.0, 0.0]))
			if args.write_attn:
				draw_attention_box(X[i, j, :, :, :], write_attn_params[k, :], np.array([0.0, 1.0, 0.0]))
			img[startr:endr, startc:endc, :] = X[i, j, :, :, :]
	return img

def xrecons_single_sample(X, read_attn_params, write_attn_params, args):
	"""
	Create a single image containing the output canvas at each of the T time
	steps.


	Parameters:
	-----------
	X :					X is a batch of image canvases. (T, batch_size, H, W, C)

	read_attn_params:	Tuple containing parameters needed to draw read
						attention box. Tuple contains: (r_cx, r_cy, r_d, r_thick)

	write_attn_params:	Tuple containing parameters needed to draw read
						attention box. Tuple contains: (r_cx, r_cy, r_d, r_thick)

	args:				Command line args.
	"""
	T, batch_size, H, W, C = X.shape

	padsize = 1
	padval = 1.0
	ph = H + 2 * padsize  # Padded height
	pw = W + 2 * padsize  # Padded width

	img = np.ones((ph, T * pw, C)) * padval

	for t in range(T):
		startr = padsize
		endr = startr + H
		startc = t * pw + padsize
		endc = startc + W
		if args.read_attn:
			draw_attention_box(X[t, args.sample_i, :, :, :], read_attn_params[t, args.sample_i, ], np.array([1.0, 0.0, 0.0]))
		if args.write_attn:
			draw_attention_box(X[t, args.sample_i, :, :, :], write_attn_params[t, args.sample_i, ], np.array([0.0, 1.0, 0.0]))
		img[startr:endr, startc:endc, :] = X[t, args.sample_i, :, :, : ]
	return img


def sigmoid(x):
	"""
	Numerically stable sigmoid function (avoids exp overflow)
	"""
	if x >= 0:
		z = np.exp(-x)
		return 1.0 / (1.0 + z)
	else:
		z = np.exp(x)
		return z / (1.0 + z)

def load_sample_file(args):
	"""
	Load samples and attention params input file.
	"""
	data_dict = np.load(args.in_files[0])
	img = data_dict['img'] # Shape: (T, batch_size, H, W, C)

	read_attn_params = None
	write_attn_params = None

	if args.read_attn:
		read_attn_params = data_dict['r_params'] # Shape: (T, num_params, batch_size)
		read_attn_params = np.swapaxes(read_attn_params, 1, 2) # Shape: (T, batch_size, num_params)
	if args.write_attn:
		write_attn_params = data_dict['w_params'] # Shape: (T, num_params, batch_size)
		write_attn_params = np.swapaxes(write_attn_params, 1, 2) # Shape: (T, batch_size, num_params)

	sigmoid_func = np.vectorize(sigmoid)

	X = sigmoid_func(img)  # x_recons=sigmoid(canvas)

	# If the image is grayscale, convert to 3-channel (RGB) so that attention
	# rectangles can be drawn for visualization:
	if X.shape[4] == 1:
		X = np.repeat(X, 3, axis=4)

	return X, read_attn_params, write_attn_params

def create_directory(dir_path):
	if not os.path.isdir(dir_path):
		os.makedirs(dir_path)

def create_10imgsx100samples(args):
	if len(args.in_files) != 1:
		sys.exit("One input file expected for '" + args.type + "', but " + str(len(args.in_files)) + " were found.")

	X, read_attn_params, write_attn_params = load_sample_file(args)
	T, batch_size, H, W, C = X.shape

	create_directory(args.out_dir)

	# Display reconstruction images
	for t in range(T):
		if args.read_attn and args.write_attn:
			img = xrecons_grid(X[t, :, :, :, :], read_attn_params[t, :, :], write_attn_params[t, :, :], args)
		elif args.read_attn:
			img = xrecons_grid(X[t, :, :, :, :], read_attn_params[t, :, :], None, args)
		elif args.write_attn:
			img = xrecons_grid(X[t, :, :, :, :], None, write_attn_params[t, :, :], args)
		else:
			img = xrecons_grid(X[t, :, :, :, :], None, None, args)
		# you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
		img_name = '%s_%d.png' % (args.prefix, t)
		img_file = os.path.join(args.out_dir, img_name)
		imsave(img_file, img)
		print(img_file)


def create_1imgx1sample(args):
	if len(args.in_files) != 1:
		sys.exit("One input file expected for '" + args.type + "', but " + str(len(args.in_files)) + " were found.")

	X, read_attn_params, write_attn_params = load_sample_file(args)
	T, batch_size, H, W, C = X.shape

	img = xrecons_single_sample(X, read_attn_params, write_attn_params, args)

	create_directory(args.out_dir)
	img_name = '%s_sample_%d.png' % (args.prefix, args.sample_i)
	img_file = os.path.join(args.out_dir, img_name)
	imsave(img_file, img)
	print(img_file)

def create_1imgx10samples(args):
	pass

def create_training_curve_plot(args):
	'''
	# Plot training loss
	f = plt.figure()
	plt.plot(Lxs, label='Reconstruction Loss Lx')
	plt.plot(Lzs, label='Latent Loss Lz')
	plt.xlabel('iterations')
	plt.legend()
	plt.savefig('%s_loss.png' % (prefix))
	'''
	pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('type', help='Type of visualization to produce. ' +
	'Options include: "10imgsx100samples", "1imgx1sample", "1imgx10samples", "train_curve"')
	parser.add_argument('out_dir', help='Path to use when storing output files.')
	parser.add_argument('in_files', nargs='+', help='Path to *.npz file(s) to read from')

	parser.add_argument('--prefix', default='out', help='Prefix to use for output file name.')
	parser.add_argument('--read_attn', action='store_true', help='Draw read attention windows on the visualization.')
	parser.add_argument('--write_attn', action='store_true', help='Draw write attention windows on the visualization.')
	parser.add_argument('--sample_i', type=int, default=0, help='Index of sample to visualize for visualizations that only process a single sample. (Default = 0)')
	args = parser.parse_args()

	if args.type == '10imgsx100samples':
		create_10imgsx100samples(args)
	elif args.type == '1imgx1sample':
		create_1imgx1sample(args)
	elif args.type == '1imgx10samples':
		create_1imgx10samples(args)
	elif args.type == 'train_curve':
		create_training_curve_plot(args)
	else:
		sys.exit("Visualization type: '" + args.type + "' not recognized.")
