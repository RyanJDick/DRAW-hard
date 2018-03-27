# takes data saved by DRAW model and generates animations
# example usage: python visualize_generated_imgs.py noattn /tmp/draw/draw_generated_images.npz

import matplotlib
import sys
import numpy as np
from scipy.misc import imsave

matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

read_attn = False
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

def xrecons_grid(X, read_attn_params, write_attn_params):
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
			if read_attn:
				draw_attention_box(X[i, j, :, :, :], read_attn_params[k, :], np.array([1.0, 0.0, 0.0]))
			if write_attn:
				draw_attention_box(X[i, j, :, :, :], write_attn_params[k, :], np.array([0.0, 1.0, 0.0]))
			img[startr:endr, startc:endc, :] = X[i, j, :, :, :]
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

if __name__ == '__main__':
	prefix = sys.argv[1]
	out_file = sys.argv[2]
	data_dict = np.load(out_file)
	img = data_dict['img']
	
	write_attn_params = data_dict['w_params'] # Shape: (T, num_params, batch_size)
	write_attn_params = np.swapaxes(write_attn_params, 1, 2) # Shape: (T, batch_size, num_params)

	T, batch_size, H, W, C = img.shape
	sigmoid_func = np.vectorize(sigmoid)

	X = sigmoid_func(img)  # x_recons=sigmoid(canvas)

	# If the image is grayscale, convert to 3-channel (RGB) so that attention
	# rectangles can be drawn for visualization:
	if C == 1:
		X = np.repeat(X, 3, axis=4)

	# Display reconstruction images
	for t in range(T):
		img = xrecons_grid(X[t, :, :, :, :], None, write_attn_params[t, :, :])
		# you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
		imgname = '%s_%d.png' % (prefix, t)
		imsave(imgname, img)
		print(imgname)
