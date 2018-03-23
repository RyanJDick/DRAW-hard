class ReadInterface:
    """
    An abstract class to be implemented by all attention readers.
    """

    _H = 0 # Height of input images
    _W = 0 # Width of input images
    _C = 0 # Number of channels in input images and in attention window
    _N = 0 # Dimension of square attention window

    def __init__(self, height, width, channels, read_n):
        """
        Initialize read attention attributes.
        """
        self._H = height
        self._W = width
        self._C = channels
        self._N = read_n

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
                            original image. Used for illustarting the read
                            attention region.

        thickness:          Thickness of line to draw when illustrating the
                            attention region. (Some attention mechanisms will
                            use a thicker line to indicate that higher variance
                            filters are being used.)
        """
        raise NotImplementedError("read(self, x, x_hat, h_dec_prev) not implemented.")
