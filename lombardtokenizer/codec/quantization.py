# Copyright (c) Meta Platforms, Inc. and affiliates.
# See licenses/README.md for the applicable third-party license.
"""Residual vector quantizer implementation."""
import math
import typing as tp
from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn

from .rvq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any
            codes that have an exponential moving average cluster size less than the
            specified threshold with randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()  # Initialize the parent class (nn.Module)

        # Store the provided arguments as instance variables
        self.n_q = n_q  # Number of quantizers
        self.dimension = dimension  # Dimension of the codebooks
        self.bins = bins  # Codebook size
        self.decay = decay  # Decay rate for exponential moving average
        self.kmeans_init = kmeans_init  # Flag to use k-means initialization
        self.kmeans_iters = kmeans_iters  # Number of k-means iterations
        self.threshold_ema_dead_code = (
            threshold_ema_dead_code  # Threshold for dead code expiration
        )

        # Initialize the residual vector quantization with the provided arguments
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: int = None,
        bandwidth: tp.Optional[float] = None,
        n_q: tp.Optional[int] = None,
        layers: tp.Optional[list] = None,
    ) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
            n_q (int): Number of quantizer used to quantize. Default: All quantizers.
            layers (list): Layers that need to return quantized. Default: None.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated number of quantizers, layer quantized outputs,
                and the associated bandwidth and any penalty term for the loss.
        """

        bw = None  # Initialize bandwidth variable
        if sample_rate:  # If sample rate is provided
            # Calculate bandwidth per quantizer
            bw_per_q = self.get_bandwidth_per_quantizer(sample_rate)
            # Calculate total bandwidth and store it as a tensor
            bw = torch.tensor(n_q * bw_per_q).to(x)
            # Determine the number of quantizers to use based on the target bandwidth
            n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        elif n_q:
            n_q = n_q  # Use the provided number of quantizers
        else:
            n_q = self.n_q  # Default to using all quantizers

        # Check if the provided layers exceed the number of quantizers
        if layers and max(layers) >= n_q:
            raise ValueError(
                f"Last layer index in layers: A {max(layers)}.\
                    Number of quantizers in RVQ: B {self.n_q}. A must be less than B."
            )

        # Perform the quantization using the residual vector quantization (RVQ)
        quantized, codes, commit_loss, quantized_list = self.vq(
            x, n_q=n_q, layers=layers
        )

        # Return the quantized result, including the quantized tensor, codes, loss,
        # quantized list, and bandwidth
        return quantized, codes, torch.mean(commit_loss), quantized_list, bw

    def get_num_quantizers_for_bandwidth(
        self, sample_rate: int, bandwidth: tp.Optional[float] = None
    ) -> int:
        """Return n_q based on specified target bandwidth."""
        bw_per_q = self.get_bandwidth_per_quantizer(
            sample_rate
        )  # Calculate bandwidth per quantizer
        n_q = self.n_q  # Default to using all quantizers

        # If a target bandwidth is provided and is greater than 0, calculate the number
        # of quantizers
        if bandwidth and bandwidth > 0.0:
            n_q = int(
                max(1, math.floor(bandwidth / bw_per_q))
            )  # Ensure at least one quantizer is used
        return n_q

    def get_bandwidth_per_quantizer(self, sample_rate: int):
        """Return bandwidth per quantizer for a given input sample rate."""
        # Calculate and return the bandwidth per quantizer based on the sample rate and
        # codebook size
        return math.log2(self.bins) * sample_rate / 1000

    def encode(
        self,
        x: torch.Tensor,
        sample_rate: int = None,
        bandwidth: tp.Optional[float] = None,
        n_q: tp.Optional[int] = None,
        st: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizers to use
        and returns indices for each quantizer.
        Args:
            x (torch.Tensor): Input tensor.
            n_q (int): Number of quantizers used to quantize. Default: All quantizers.
            st (int): Start to encode input from which layers. Default: 0.
        """

        if sample_rate:  # If sample rate is provided
            # Determine the number of quantizers based on the target bandwidth
            n_q = self.get_num_quantizers_for_bandwidth(sample_rate, bandwidth)
        elif n_q:
            n_q = n_q  # Use the provided number of quantizers
        else:
            n_q = self.n_q  # Default to using all quantizers

        st = st or 0  # Set the starting layer to 0 if not provided

        # Encode the input tensor using RVQ and return the indices
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        Args:
            codes (torch.Tensor): Input indices for each quantizer.
            st (int): Start to decode input codes from which layers. Default: 0.
        """
        # Decode the indices back to the quantized representation using RVQ
        quantized = self.vq.decode(codes, st=st)
        return quantized

