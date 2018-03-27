import tensorflow as tf
from .read_interface import ReadInterface
from .write_interface import WriteInterface

eps = 1e-8  # epsilon for numerical stability

def filterbank(gx, gy, sigma2, delta, H, W, N):
	"""
	Helper function used by both ReadSoftAttn and WriteSoftAttn.

	Create gaussian attention filters Fx and Fy based on the parameters provided.
	See the paper for a complete explanation of how these Gaussian filters are
	generated.

	Parameters:
	----------
	gx :       x-ccordinate of the center of the grid of attention filters

	gy :       y-coordinate of the cnter of the frid of attention filter

	sigma2 :   variance of the gaussian filters to use. Larger variance means
			   more contribution from pixels far from the filter center. As
			   variance decreases, it approaches a 'hard' attention approach.

	delta : 	Spacing between filter centers.

	H, W :		Height and Width of input image.

	N : 		Dimension of the square attention window in pixels. There are
				N x N gaussian filters used to generate the attention window.

	Return:
	-------
	Fx, Fyx and y axis Gaussian attention filters
	"""
	grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
	mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
	mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
	a = tf.reshape(tf.cast(tf.range(W), tf.float32), [1, 1, -1])
	b = tf.reshape(tf.cast(tf.range(H), tf.float32), [1, 1, -1])
	mu_x = tf.reshape(mu_x, [-1, N, 1])
	mu_y = tf.reshape(mu_y, [-1, N, 1])
	sigma2 = tf.reshape(sigma2, [-1, 1, 1])
	Fx = tf.exp(-tf.square(a - mu_x) / (2 * sigma2))
	Fy = tf.exp(-tf.square(b - mu_y) / (2 * sigma2)) # batch x N x H
	# normalize, sum over W and H dims
	Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), eps)
	Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), eps)
	return Fx, Fy


def attn_window(scope, h_dec, H, W, N):
	"""
	Helper function used by both ReadSoftAttn and WriteSoftAttn.

	Predicts the appropriate attention parameters, and returns the resultant
	filters.

	Parameters
	----------
	scope :     Name of scope to use for tensorflow variables. This is necessary
				to avoid sharing weights between the fully connected layers used
				to predict the read attention parameters and the write attention
				parameters.

	h_dec :     Output from decoder. This tensor has shape (B, decoder_output_size)

	H, W :		Height of and width of input images.
	N :         Dimension of the square attention window in pixels. There are N x N
				gaussian filters used to generate the attention window.

	Return
	------
	(Fx, Fy, gamma, [cx, cy, d, thickness])

	Fx, Fy:     x and y axis Gaussian attention filters produced by
				filterbank(gx, gy, sigma2, delta, N).

	gamma:      Attenuation parameter.

	cx, cy, d, thickness:
				Parameters that are used to illustrate the attention window. See
				defintions of each in the docstring for
				ReadSoftAttn.read(self, x, x_hat, h_dec_prev)

	"""
	with tf.variable_scope(scope):
		## Apply linear fully connected layer to predict attention parameters:
		params = tf.contrib.layers.fully_connected(h_dec, 5, activation_fn=None, scope='fc')
		gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
		gx = (W + 1) / 2 * (gx_ + 1)
		gy = (H + 1) / 2 * (gy_ + 1)
		sigma2 = tf.exp(log_sigma2)
		gamma = tf.exp(log_gamma)
		delta = (max(W, H) - 1) / (N - 1) * tf.exp(log_delta) # batch x N

		## Construct Gaussian attention filters given parameters:
		Fx, Fy = filterbank(gx, gy, sigma2, delta, H, W, N)

		## Determine parameters for illustrating the attention window
		cx = gx
		cy = gy
		d = (N - 1) * delta
		# Thickness of illustrated attention regions varies linearly with filter variance
		thickness = 1.0 * sigma2

		return Fx, Fy, gamma, [cx, cy, d, thickness]


class ReadSoftAttn(ReadInterface):
	"""
	A class for reading with 'soft' attention as was implmented in the original
	DRAW paper.
	This class implements the ReadInterface.
	"""

	# __init__ is inherited from ReadInterface

	def _filter_img(self, img, Fx, Fy, gamma):
		"""
		Apply attention filters to image.

		Parameters
		----------
		img:        Image to apply the filters to. Should have dimensions
					(B, H, W, 1)

		Fx, Fy:     x and y-axis attention filters

		gamma:      attenuation constant

		Returns
		-------
		glimpse:    Attention window. (B, N, N, 1)
		"""
		Fxt = tf.transpose(Fx, perm = [0, 2, 1])
		img = tf.squeeze(img, axis=-1) #(B, H, W)
		glimpse = tf.matmul(Fy, tf.matmul(img, Fxt))
		glimpse = tf.reshape(glimpse, [-1, self._N * self._N])
		glimpse = glimpse * tf.reshape(gamma, [-1, 1])
		glimpse = tf.reshape(glimpse, [-1, self._N, self._N, 1])
		return glimpse


	def read(self, x, x_hat, h_dec_prev):
		"""
		Implements the read attention mechanism.

		Parameters
		----------
		x:                  Input image tensor. Pixel values range between 0 and
							1. Image has shape (B, _H, _W, _C).

		x_hat:              Error image tensor obtained by subtracting current
							reconstruction canvas from the input image. Again,
							this tensor has shape (B, _H, _W, _C).

		h_dec_prev:         Output from the decoder in the previous time step.
							Tensor has shape (B, decoder_output_size).

		Return
		------
		(attention_window, [cx, cy, d, thickness])

		attention_window :  Tensor containing the attention window. Has shape of
							(B, _N x _N x C x 2)

		cx, cy:             The x and y coordinates of the center point of the
							attention window for illustration purposes.

		d:                  The dimension of the square attention window in the
							original image. Used for illustarting the read
							attention region.

		thickness:          Thickness of line to draw when illustrating the
							attention region. (Some attention mechanisms will
							use a thicker line to indicate that higher variance
							filters are being used.)
		"""
		with tf.variable_scope('read'):
			## Generate attention window:
			Fx, Fy, gamma, visualization_params = attn_window("soft_attn", h_dec_prev, self._H, self._W, self._N)
			x_combined = tf.concat([x, x_hat], -1)
			# Split channels to obtain array of 2 * C tensors, where each tensor has
			# dimensions (B, H, W, 1):
			x_combined_channels = tf.split(x_combined, 2 * self._C, axis=-1)
			x_combined_glimpse_channels = []
			for x_combined_channel in x_combined_channels:
				x_combined_glimpse_channels.append(self._filter_img(x_combined_channel, Fx, Fy, gamma))
			# Concatenate the attention windows from all channels to obtain a tensor
			# of shape (B, N x N x C x 2)
			x_combined_glimpse = tf.concat(x_combined_glimpse_channels, -1)
			x_combined_glimpse = tf.reshape(x_combined_glimpse, [-1, self._N * self._N * self._C * 2])
			return x_combined_glimpse, visualization_params

class WriteSoftAttn(WriteInterface):
	"""
	A class for writing with 'soft' attention as was implmented in the original
	DRAW paper.
	This class implements the WriteInterface.
	"""

	# __init__ is inherited from WriteInterface

	def write(self, h_dec):
		"""
		Implements the write attention mechanism.

		Parameters
		----------
		h_dec:              Output from the decoder in the current time step.
							Tensor has shape (B, decoder_output_size).

		Return
		------
		(attention_window, [cx, cy, d, thickness])

		write_canvas :      Tensor containing the update to be added to the
							reconstruction canvas. Attention is used to generate
							this write canvas. This tensor has a shape of
							(B, _H, _W, _C).

		cx, cy:             The x and y coordinates of the center point of the
							attention window for illustration purposes.

		d:                  The dimension of the square attention window in the
							original image. Used for illustrating the write
							attention region.

		thickness:          Thickness of line to draw when illustrating the
							attention region. (Some attention mechanisms will
							use a thicker line to indicate that higher variance
							filters are being used.)
		"""
		## Apply fully connected linear layer to generate write canvas
		with tf.variable_scope('write'):
			## Predict 'what' to write
			w = self._generate_write_patch(h_dec)

			## Determine 'where' to write
			Fx, Fy, gamma, visualization_params = attn_window("soft_attn", h_dec, self._H, self._W, self._N)
			Fyt = tf.transpose(Fy, perm=[0, 2, 1])
			w_channels = tf.split(w, self._C, axis=-1)
			wr_channels = []
			for w_channel in w_channels:
				wr = tf.matmul(Fyt, tf.matmul(tf.squeeze(w_channel, axis=-1), Fx))
				wr = tf.reshape(wr, [-1, self._H * self._W])
				wr = wr * tf.reshape(1.0 / gamma, [-1, 1])
				wr = tf.reshape(wr, [-1, self._H, self._W, 1])
				wr_channels.append(wr)
			wr = tf.concat(wr_channels, -1)
			return wr, visualization_params
