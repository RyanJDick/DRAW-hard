import tensorflow as tf
from .read_interface import ReadInterface
from .write_interface import WriteInterface
from .transformer import transformer

def spatial_transformer_attn_window_params(scope, h_dec):
	'''
	Helper function used by both ReadSpatialTransformerAttn and
	WriteSpatialTransformerAttn.

	Apply a localization network with a single hidden layer to the h_dec output
	in order to determine the parameters used to define the attention window.
	The parameters to be predicted are:
	s - scaling parameter
	tx - x translation
	ty - y translation

	These parameters are used to define an affine transformation to be applied
	by a spatial transformer network as follows:
	A = [[s, 0, tx],
		 [0, s, ty]]

	The fully-connected layer is initialized to achieved an identity affine
	transformation (no transformation) before training:
	A0 = [[1, 0, 0],
		  [0, 1, 0]]
	'''
	# Define 'localization network':
	with tf.variable_scope(scope):
		with tf.variable_scope('scale'):
			scale = tf.contrib.layers.fully_connected(h_dec, 1,
				#activation_fn=tf.nn.sigmoid, # Limit scale to (0,1)
				weights_initializer=tf.zeros_initializer,
				biases_initializer=tf.ones_initializer, # Initially, output scale = 1
				scope='fc')
			s = scale[:, 0]
		with tf.variable_scope('shift'):
			shift = tf.contrib.layers.fully_connected(h_dec, 2,
				#activation_fn=tf.nn.tanh, # Limit translation to (-1, 1)
				weights_initializer=tf.zeros_initializer,
				biases_initializer=tf.zeros_initializer, # Initially, output shift = 0
				scope='fc')
			tx, ty = shift[:, 0], shift[:, 1]
		return s, tx, ty

def params_to_transformation_matrix(s, tx, ty):
	"""
	Helper function used by both ReadSpatialTransformerAttn and
	WriteSpatialTransformerAttn.

	Combine scale and translation into transformation matrix

	This operation produces the following tensor (for a batch size of 2):
	[[[s1,  0, tx1],  # batch 1
	  [ 0, s1, ty1]],
	 [[s2,  0, tx2],  # batch 2
	  [ 0, s2, ty2]]]
	"""
	return tf.stack([
		tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(tx, 1)], axis=1),
		tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(ty, 1)], axis=1),
		], axis=1)

class ReadSpatialTransformerAttn(ReadInterface):
	"""
	A class for reading with spatial transformer attention.
	This class implements the ReadInterface.
	"""

	# __init__ is inherited from ReadInterface

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
			s, tx, ty = spatial_transformer_attn_window_params('spatial_transformer_attn_params', h_dec_prev)
			transformation_mat = params_to_transformation_matrix(s, tx, ty)
			with tf.variable_scope('spatial_transformer_network'):
				# Full canvas to attention window transformation
				x_combined = tf.concat([x, x_hat], -1)
				x_window = transformer(x_combined, transformation_mat, [self._N, self._N])
				x_window = tf.reshape(x_window, [-1, self._N * self._N * self._C * 2])

			## Determine attention visualization parameters
			batch_size = x.get_shape()[0]
			cx = (self._W / 2) + tx * (self._W / 2)
			cy = (self._H / 2) + ty * (self._H / 2)
			d = s * self._W # Assumes that image is a square
			thickness = tf.constant(1.0, shape=[batch_size])

			return x_window, [cx, cy, d, thickness]


class WriteSpatialTransformerAttn(WriteInterface):
	"""
	A class for writing with spatial transformer attention.
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
			s, tx, ty = spatial_transformer_attn_window_params('spatial_transformer_attn_params', h_dec)
			# 'Invert' transformation parameters because we are now going from
			# window to canvas, rather than the reverse:
			s_write = 1.0 / s
			tx_write = -tx / s
			ty_write = -ty / s
			transformation_mat = params_to_transformation_matrix(s_write, tx_write, ty_write)

			with tf.variable_scope('spatial_transformer_network'):
				canvas = transformer(w, transformation_mat, [self._H, self._W])

			## Determine attention visualization parameters
			batch_size = w.get_shape()[0]
			cx = (self._W / 2) + tx * (self._W / 2)
			cy = (self._H / 2) + ty * (self._H / 2)
			d = s * self._W # Assumes that image is a square
			thickness = tf.constant(1.0, shape=[batch_size])

			return canvas, [cx, cy, d, thickness]
