class WriteInterface:
    """
    An abstract class to be implemented by all attention writers.
    """

    self._H = 0 # Height of input images
    self._W = 0 # Width of input images
    self._C = 0 # Number of channels in input images and in attention window
    self._N = 0 # Dimension of square attention window

    def __init__(self, height, width, channels, read_n):
        """
        Initialize write attention attributes.
        """
        self._H = height
        self._W = width
        self._C = channels
        self._N = read_n

    def write(self, h_dec):
        """
        Implements the write attention mechanism.

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
        raise NotImplementedError("write(self, h_dec) not implemented.")
