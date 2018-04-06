import tensorflow as tf
from .write_interface import WriteInterface

class WriteStochasticAttn(WriteInterface):
	"""
	A class for writing with stochastic attention.
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
		(attention_window, [cx, cy, d, thickness], [alphas, samples])

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

        ######################################################################
        ### The following return values are unique to stochastic attention ###

        dist:				The one-hot categorical distribution from which the
							alpha sample was drawn.

        samples:            The sampled attention location.
		"""
		## Apply fully connected linear layer to generate write canvas
		with tf.variable_scope('write'):
			## Predict 'what' to write
			w = self._generate_write_patch(h_dec)

			## Determine 'where' to write
			if not (self._H % self._N == 0 and self._W % self._N == 0):
                exit("Stochastic attention: Expected write_n to evenly divide image dimensions.")
            # The possible attention regions are created by first overlaying a
            # grid on the canvas, where each grid cell has dimensions N x N
            # Larger attention regions (2N x 2N) are then also placed with
            # center points at all of the intersections created by the small grid

            # smallCols and smallRows refer to the pixel coordinates of the
            # upper left corner of the small attention regions
            small_cols = range(0, self._W, self._N)
            small_rows = range(0, self._H, self._N)
            big_cols = small_cols[:-1]
            big_rows = small_rows[:-1]

            num_attn_regions = (len(small_cols) * len(small_rows)) + (len(big_cols) * len(big_rows))

			with tf.variable_scope('stochastic_sampling_params'):
				with tf.variable_scope('alphas'):
        			alpha_logits = tf.contrib.layers.fully_connected(h_dec, num_attn_regions,
						activation_fn=None, # Generate logits in range (-inf, inf)
        				weights_initializer=tf.zeros_initializer, # Initially, all regions are equally likely
        				scope='fc')

				with tf.variable_scope('sample'):
					dist = tf.contrib.distributions.OneHotCategorical(logits=alpha_logits)

            alpha_sample = dist.sample() # Draw a single one-hot sample for each image in batch (batch_size, num_attn_regions)

			attn_windows = []
			cxs = []
			cys = []
			ds = []

			# Add small attention windows to list of windows:
			for small_col in small_cols:
				for small_row in small_rows:
					paddings = tf.constant([[0, 0],
											[small_row, self._H - small_row - self._N],  # [[top, bottom],
											[small_col, self._W - small_col - self._N],  #  [left, right]]
											[0, 0]])
					attn_windows.append(tf.pad(w, paddings))
					cxs.append(small_col + (self._N / 2))
					cys.append(small_row + (self._N / 2))
					ds.append(self._N)

			# Add large attention windows to list of windows:
			for big_col in big_cols:
				for big_row in big_rows:
					paddings = tf.constant([[0, 0],
											[big_row, self._H - big_row - (2 * self._N)],  # [[top, bottom],
											[big_col, self._W - big_col - (2 * self._N)],  #  [left, right]]
											[0, 0]])
					attn_windows.append(tf.pad(w, paddings))
					cxs.append(big_col + self._N)
					cys.append(big_row + self._N)
					ds.append(self._N)

			attn_windows = tf.stack(attn_windows, axis=1) # Shape: (batch_size, num_attn_regions, _H, _W, _C)

			## Multiply attn windows by alpha_sample:

			# Add three dimensions to shape of alpha_sample so it broadcasts
			# properly in multiplication with attn_windows
			for _ in range(3):
				alpha_sample_expanded = tf.expand_dims(alpha_sample, axis=-1)

			attn_contributions = tf.multiply(attn_windows, alpha_sample_expanded)
			canvas = tf.reduce_sum(attn_contributions, axis=1) # Shape: (batch_size, _H, _W, _C)

			## Determine attention visualization parameters
			batch_size = w.get_shape()[0]
			index = tf.expand_dims(tf.argmax(alpha_sample, axis=-1), axis=-1)
			cxs = tf.constant(cxs)
			cx = tf.gather_nd(cxs, index)

			cys = tf.constant(cys)
			cy = tf.gather_nd(cys, index)

			ds = tf.constant(ds)
			d =  tf.gather_nd(ds, index)
			thickness = tf.constant(1.0, shape=[batch_size])

			return canvas, [cx, cy, d, thickness], dist, alpha_sample
