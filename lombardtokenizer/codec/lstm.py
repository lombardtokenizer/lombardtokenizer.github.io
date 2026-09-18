# EnCodec-derived building block; see licenses/README.md.
from torch import nn


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.dimension = dimension
        self.num_layers = num_layers
        self.skip = skip
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            dimension, dimension, num_layers, bidirectional=bidirectional
        )

    def forward(self, x):
        """
        Forward pass for the SLSTM layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, input_size, seq_len).
        """
        # Permute to (seq_len, batch_size, input_size)
        x = x.permute(2, 0, 1)

        # Pass through LSTM
        y, _ = self.lstm(x)

        # Handle bidirectional case
        if self.bidirectional:
            # Concatenate forward and backward states
            x = x.repeat(1, 1, 2)

        # Apply skip connection if enabled
        if self.skip:
            y = y + x

        # Permute back to (batch_size, input_size, seq_len)
        y = y.permute(1, 2, 0)
        return y

