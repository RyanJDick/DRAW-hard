from write_interface import WriteInterface

class WriteNoAttn(WriteInterface):
    """
    A class for writing without attention (simply reading the entire image).
    This class implements the WriteInterface.
    """

    # __init__ is inherited from WriteInterface

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
        ## Apply fully connected linear layer to generate write canvas
        write_size = self._H * self._W * self._C
        w = tf.contrib.layers.fully_connected(h_dec, write_size, activation_fn=None, scope='write_no_attn_fc')
        w = tf.reshape(w, [-1, self._H, self._W, self._C])

        ## ## Determine appropriate parameters for attention visualization:
        cx = (self._W + 1) / 2
        cy = (self._H + 1) / 2
        d = min(self._H, self._W) # This is going to be wrong if the image isn't square
        thickness = 1.0 # Arbitrary value to be updated

        return w, cx, cy, d, thickness
