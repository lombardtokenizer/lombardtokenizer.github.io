# Copyright (c) Meta Platforms, Inc. and affiliates.
# See licenses/README.md for the applicable third-party license.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""
import typing as tp

from einops import rearrange, repeat, reduce, einsum
import torch
from torch import nn
import torch.nn.functional as F



def default(val: tp.Any, d: tp.Any) -> tp.Any:
    # Returns `val` if not None, otherwise returns `d`.
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    # Exponential Moving Average (EMA) update in-place.
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    # Applies Laplace smoothing to a tensor for stability in computations involving
    # categorical distributions.
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    # Initializes a tensor using Kaiming uniform distribution.
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    # Samples a number of vectors from the given tensor.
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        # Random permutation if sufficient samples are available
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        # Otherwise, randomly sample with replacement
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack(
        [sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0
    )


def cdist(x, y):
    x2 = reduce(x**2, "n d -> n", "sum")
    y2 = reduce(y**2, "c d -> c", "sum")
    xy = einsum(x, y, "n d, c d -> n c") * -2
    return (
        (rearrange(x2, "n -> n 1") + rearrange(y2, "c -> 1 c") + xy).clamp(min=0).sqrt()
    )


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    # Performs k-means clustering to initialize centroids.
    dim, dtype = samples.shape[-1], samples.dtype
    # Initialize cluster means by sampling vectors
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        # Calculate Euclidean distance between samples and cluster centers
        # diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        # dists = -(diffs**2).sum(dim=-1)
        dists = cdist(samples, means)
        # Assign each sample to the closest cluster
        buckets = dists.min(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        # Calculate new means based on assigned samples
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        # Update means, keeping previous means for empty clusters
        means = torch.where(zero_mask[..., None], means, new_means)
    # Return final cluster centers and their sizes
    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,  # Dimension of the code vectors
        codebook_size: int,  # Number of code vectors in the codebook
        kmeans_init: int = False,  # Whether to use k-means for codebook initialization
        kmeans_iters: int = 10,  # Number of iterations for k-means if used
        decay: float = 0.99,  # Decay factor for the exponential moving average (EMA)
        epsilon: float = 1e-5,  # Small value for numerical stability
        threshold_ema_dead_code: int = 2,  # Threshold for identifying and replacing "dead" codes
    ):
        super().__init__()  # Initialize the parent class (nn.Module)
        self.decay = decay  # Store the decay factor

        # Choose the initialization function: uniform random if not using k-means,
        # otherwise zeros
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)  # Initialize the codebook embedding matrix

        self.codebook_size = codebook_size  # Store the codebook size
        self.kmeans_iters = kmeans_iters  # Store the number of k-means iterations
        self.epsilon = epsilon  # Store the epsilon value for numerical stability
        self.threshold_ema_dead_code = (
            threshold_ema_dead_code  # Store the threshold for dead codes
        )

        # Register buffer to track if the codebook has been initialized
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        # Register buffer to track the size of each cluster in the codebook
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        # Register buffer for the actual codebook vectors
        self.register_buffer("embed", embed)
        # Register buffer for the average of codebook vectors, used for EMA updates
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        # Initialize the codebook using k-means if it hasn't been initialized yet
        if self.inited:
            return  # If already initialized, do nothing

        # Run k-means clustering on the data to initialize the codebook
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        # Copy the k-means centroids into the codebook embedding matrix
        self.embed.data.copy_(embed)
        # Copy the centroids into the EMA buffer
        self.embed_avg.data.copy_(embed.clone())
        # Store the cluster sizes from k-means
        self.cluster_size.data.copy_(cluster_size)
        # Mark the codebook as initialized
        self.inited.data.copy_(torch.Tensor([True]))

    def replace_(self, samples, mask):
        # Replace expired codes with randomly sampled vectors from the current batch
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        # Update the codebook with the modified codebook
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        # If the threshold for dead codes is zero, do nothing
        if self.threshold_ema_dead_code == 0:
            return

        # Identify the codes that have a cluster size below the threshold
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return  # If no expired codes, do nothing

        # Flatten the batch samples for processing
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        # Replace the expired codes with new samples from the batch
        self.replace_(batch_samples, mask=expired_codes)

    def preprocess(self, x):
        # Flatten the input tensor for processing
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        # Compute the distances between input vectors and codebook vectors
        embed = self.embed.t()  # Transpose the codebook matrix
        dist = -(
            x.pow(2).sum(
                1, keepdim=True
            )  # Compute the squared L2 norm of the input vectors
            - 2
            * x
            @ embed  # Compute the dot product between input and codebook vectors
            + embed.pow(2).sum(
                0, keepdim=True
            )  # Compute the squared L2 norm of the codebook vectors
        )
        embed_ind = dist.max(
            dim=-1
        ).indices  # Get the indices of the closest codebook vectors
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        # Reshape the indices back to the original input shape
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        # Map the quantized indices back to the corresponding codebook vectors
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape  # Store the original shape of the input
        x = self.preprocess(x)  # Preprocess the input by flattening it
        embed_ind = self.quantize(
            x
        )  # Quantize the input to get the nearest codebook indices
        embed_ind = self.postprocess_emb(
            embed_ind, shape
        )  # Reshape the indices back to the original input shape
        return embed_ind

    def decode(self, embed_ind):
        # Dequantize the indices back to the original vectors
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype  # Store the shape and data type of the input
        x = self.preprocess(x)  # Preprocess the input by flattening it
        self.init_embed_(x)  # Initialize the codebook if not already initialized
        embed_ind = self.quantize(
            x
        )  # Quantize the input to get the nearest codebook indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(
            dtype
        )  # Convert the indices to one-hot encoding
        embed_ind = self.postprocess_emb(
            embed_ind, shape
        )  # Reshape the indices back to the original input shape
        quantize = self.dequantize(
            embed_ind
        )  # Dequantize the indices back to the original vectors
        if self.training:
            # During training, handle codebook updates and code expiration
            self.expire_codes_(x)  # Check for and replace expired codes
            # Update the cluster sizes using EMA
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = (
                x.t() @ embed_onehot
            )  # Sum the input vectors associated with each codebook vector
            ema_inplace(
                self.embed_avg, embed_sum.t(), self.decay
            )  # Update the EMA of the codebook vectors
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )  # Apply Laplace smoothing to the cluster sizes for numerical stability
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(
                1
            )  # Normalize the EMA of the codebook vectors
            self.embed.data.copy_(
                embed_normalized
            )  # Update the codebook with the normalized vectors

        return (
            quantize,
            embed_ind,
        )  # Return the quantized vectors and their corresponding indices


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    A PyTorch module for vector quantization that supports Euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified
            dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any
            codes that have an exponential moving average cluster size less than the
            specified threshold with randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,  # Dimension of the input vectors
        codebook_size: int,  # Number of vectors in the codebook
        codebook_dim: tp.Optional[
            int
        ] = None,  # Dimension of the codebook vectors, defaults to dim if not provided
        decay: float = 0.99,  # Decay factor for the exponential moving average (EMA)
        epsilon: float = 1e-5,  # Small value for numerical stability
        kmeans_init: bool = True,  # Whether to use k-means for codebook initialization
        kmeans_iters: int = 50,  # Number of iterations for k-means if used
        threshold_ema_dead_code: int = 2,  # Threshold for identifying and replacing "dead" codes
        commitment_weight: float = 1.0,  # Weight for the commitment loss term
    ):
        super().__init__()  # Initialize the parent class (nn.Module)

        _codebook_dim: int = default(
            codebook_dim, dim
        )  # Set codebook dimension to dim if not provided

        # Determine if input projection is needed (if input dim is different from codebook dim)
        requires_projection = _codebook_dim != dim
        # Define input projection layer if needed, else use identity (no change)
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        # Define output projection layer if needed, else use identity (no change)
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon  # Store the epsilon value for numerical stability
        self.commitment_weight = (
            commitment_weight  # Store the weight for the commitment loss
        )

        # Initialize the EuclideanCodebook with the specified parameters
        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,  # Codebook dimension
            codebook_size=codebook_size,  # Codebook size
            kmeans_init=kmeans_init,  # Use k-means for initialization if specified
            kmeans_iters=kmeans_iters,  # Number of k-means iterations
            decay=decay,  # Decay factor for EMA
            epsilon=epsilon,  # Epsilon for numerical stability
            threshold_ema_dead_code=threshold_ema_dead_code,  # Threshold for dead code replacement
        )
        self.codebook_size = codebook_size  # Store the codebook size

    @property
    def codebook(self):
        # Return the current codebook embeddings
        return self._codebook.embed

    def encode(self, x):
        # Reshape input from (batch_size, dim, n) to (batch_size, n, dim)
        x = rearrange(x, "b d n -> b n d")
        # Project input to the codebook dimension if needed
        x = self.project_in(x)
        # Encode the input using the codebook (returns nearest code indices)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        # Decode quantized indices back to codebook vectors
        quantize = self._codebook.decode(embed_ind)
        # Project back to the original input dimension if needed
        quantize = self.project_out(quantize)
        # Reshape from (batch_size, n, dim) to (batch_size, dim, n)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device  # Get the device of the input tensor (e.g., CPU or GPU)
        # Reshape input from (batch_size, dim, n) to (batch_size, n, dim)
        x = rearrange(x, "b d n -> b n d")
        # Project input to the codebook dimension if needed
        x = self.project_in(x)

        # Quantize the input using the codebook, obtaining quantized vectors and indices
        quantize, embed_ind = self._codebook(x)

        if self.training:
            # During training, enforce commitment to the quantized code
            quantize = x + (quantize - x).detach()

        # Initialize loss tensor, required for autograd during training
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            # Compute commitment loss if the weight is greater than zero
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(
                    quantize.detach(), x
                )  # Mean squared error loss
                loss = (
                    loss + commit_loss * self.commitment_weight
                )  # Scale by commitment weight

        # Project quantized vectors back to the original input dimension if needed
        quantize = self.project_out(quantize)
        # Reshape from (batch_size, n, dim) to (batch_size, dim, n)
        quantize = rearrange(quantize, "b n d -> b d n")

        # Return the quantized vectors, their indices, and the loss (if any)
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Implementation of Residual Vector Quantization (RVQ), which quantizes residuals in
    multiple stages. Follows Algorithm 1 in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()  # Initialize the parent class (nn.Module)
        # Create a list of VectorQuantization layers, one for each quantizer
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(
        self, x, n_q: tp.Optional[int] = None, layers: tp.Optional[list] = None
    ):
        quantized_out = 0.0  # Initialize the output quantized tensor to zero
        residual = x  # Set the initial residual to the input tensor

        all_losses = []  # List to store losses from each quantization stage
        all_indices = []  # List to store the indices of the quantized vectors
        out_quantized = (
            []
        )  # List to store quantized outputs if specific layers are provided

        n_q = n_q or len(
            self.layers
        )  # Set the number of quantization stages (n_q) to all layers if not provided

        # Iterate over the layers, up to the specified number of quantizers
        for i, layer in enumerate(self.layers[:n_q]):
            # Apply the current quantization layer to the residual
            quantized, indices, loss = layer(residual)
            residual = (
                residual - quantized
            )  # Update the residual by subtracting the quantized output
            quantized_out = quantized_out + quantized  # Accumulate the quantized output

            # Store the indices and loss for the current quantization stage
            all_indices.append(indices)
            all_losses.append(loss)

            # If specific layers are provided and the current layer is in the list, store the quantized output
            if layers and i in layers:
                out_quantized.append(quantized)

        # Stack all collected losses and indices into tensors
        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))

        # Return the accumulated quantized output, stacked indices, losses, and any specified layer outputs
        return quantized_out, out_indices, out_losses, out_quantized

    def encode(
        self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None
    ) -> torch.Tensor:
        residual = x  # Set the initial residual to the input tensor
        all_indices = []  # List to store the indices of the quantized vectors
        n_q = n_q or len(
            self.layers
        )  # Set the number of quantization stages to all layers if not provided
        st = st or 0  # Set the starting index to 0 if not provided

        # Iterate over the layers from the starting index up to the specified number of quantizers
        for layer in self.layers[st:n_q]:
            indices = layer.encode(residual)  # Encode the residual and get the indices
            quantized = layer.decode(
                indices
            )  # Decode the indices back to quantized vectors
            residual = (
                residual - quantized
            )  # Update the residual by subtracting the quantized output
            all_indices.append(
                indices
            )  # Store the indices for the current quantization stage

        # Stack all collected indices into a tensor
        out_indices = torch.stack(all_indices)

        # Return the stacked indices
        return out_indices

    def decode(self, q_indices: torch.Tensor, st: int = 0) -> torch.Tensor:
        # Initialize the output quantized tensor to zero on the same device as the indices
        quantized_out = torch.tensor(0.0, device=q_indices.device)

        # Iterate over the quantization indices and corresponding layers
        for i, indices in enumerate(q_indices):
            layer = self.layers[st + i]  # Get the corresponding layer
            quantized = layer.decode(
                indices
            )  # Decode the indices back to quantized vectors
            quantized_out = quantized_out + quantized  # Accumulate the quantized output

        # Return the accumulated quantized output
        return quantized_out
