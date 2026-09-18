"""Codec components required by LombardTokenizer.

The implementations in this package are the only runtime codec dependency of
the public LombardTokenizer model.  They preserve the original third-party
license headers in the individual modules.
"""

from .norm import ConvLayerNorm
from .lstm import SLSTM
from .conv import NormConv1d, NormConv2d, NormConvTranspose1d, NormConvTranspose2d
from .conv import SConv1d, SConvTranspose1d, pad1d, unpad1d
from .discriminators import (
    CodecDiscriminator,
    DiscriminatorOutput,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator,
)
from .seanet import SEANetDecoder, SEANetEncoder
from .quantization import QuantizedResult, ResidualVectorQuantizer
from .rvq import ResidualVectorQuantization, VectorQuantization

__all__ = [
    "ConvLayerNorm",
    "CodecDiscriminator",
    "DiscriminatorOutput",
    "NormConv1d",
    "NormConv2d",
    "NormConvTranspose1d",
    "NormConvTranspose2d",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "MultiScaleSTFTDiscriminator",
    "QuantizedResult",
    "ResidualVectorQuantization",
    "ResidualVectorQuantizer",
    "SConv1d",
    "SConvTranspose1d",
    "SEANetDecoder",
    "SEANetEncoder",
    "SLSTM",
    "VectorQuantization",
    "pad1d",
    "unpad1d",
]
