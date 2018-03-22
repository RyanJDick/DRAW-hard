from read_interface import ReadInterface
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
    	x_combined_channels = tf.split(x_combined, 2 * self._C, axis=-1) # 2*C tensors of shape (B, H, W, 1)
    	read_features = []
    	for x_combined_channel in x_combined_channels:
    		read_features.append(tf.reshape(x_combined_channel, [batch_size, self._H * self._W]))
    	read_features = tf.concat(read_features, -1) # (B, H * W * C * 2)

        ## Determine appropriate parameters for attention visualization:
        cx = (self._W + 1) / 2
        cy = (self._H + 1) / 2
        d = min(self._H, self._W) # This is going to be wrong if the image isn't square
        thickness = 1.0 # Arbitrary value to be updated
        
    	return read_features, cx, cy, d, thickness
