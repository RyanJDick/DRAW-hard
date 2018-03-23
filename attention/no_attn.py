from read_interface import ReadInterface
from write_interface import WriteInterface

class ReadNoAttn(ReadInterface):
    """
    A class for reading without attention (simply reading the entire image).
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
        (attention_window, cx, cy, d, thickness)

        attention_window :  Tensor containing the attention window. Has shape of
                            (B, _N x _N x C x 2)

        cx, cy:             The x and y coordinates of the center point of the
                            attention window for illustration purposes.

        d:                  The dimension of the square attention window in the
                            original image. Used for illustrating the read
                            attention region.

        thickness:          Thickness of line to draw when illustrating the
                            attention region. (Some attention mechanisms will
                            use a thicker line to indicate that higher variance
                            filters are being used.)
        """
        ## Reshape x and x_hat to desired output dimensions:
        x_combined = tf.concat([x, x_hat], -1) # (B, H, W, 2 * C)
    	read_features = tf.reshape(x_combined, [-1, self._H * self._W * self._C * 2])

        ## Determine appropriate parameters for attention visualization:
        cx = (self._W + 1) / 2
        cy = (self._H + 1) / 2
        d = min(self._H, self._W) # This is going to be wrong if the image isn't square
        thickness = 1.0 # Arbitrary value to be updated

    	return read_features, cx, cy, d, thickness

class WriteNoAttn(WriteInterface):
    """
    A class for writing without attention (simply reading the entire image).
    This class implements the WriteInterface.
    """

    # __init__ is inherited from WriteInterface

    def _generate_write_patch(h_dec):
        """
        Overrides implementation in WriteInterface in order to generate a write
        patch the same size of the input image.

        Applies a fully connected linear layer to generate the attention patch
        to be written.

        Parameters
        ----------
        h_dec:      Output of decoder. (B x decoder_output_size)

        Return
        ------
        w:          Write patch. (B, H, W, C)
        """
        with tf.variable_scope('w_patch')
            write_size = self._H * self._W * self._C
            w = tf.contrib.layers.fully_connected(h_dec, write_size, activation_fn=None, scope='fc')
            w = tf.reshape(w, [-1, self._H, self._W, self._C])
            return w


    def write(self, h_dec):
        """
        Implements the write step with no attention.

        Parameters
        ----------
        h_dec:              Output from the decoder in the current time step.
                            Tensor has shape (B, decoder_output_size).

        Return
        ------
        (attention_window, cx, cy, d, thickness)

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

        with tf.variable_scope('write'):
            ## Predict 'what' to write
            w = self._generate_write_patch(h_dec)

            ## Determine appropriate parameters for attention visualization:
            cx = (self._W + 1) / 2
            cy = (self._H + 1) / 2
            d = min(self._H, self._W) # This is going to be wrong if the image isn't square
            thickness = 1.0 # Arbitrary value to be updated

            return w, cx, cy, d, thickness
