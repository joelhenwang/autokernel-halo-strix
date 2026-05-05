"""Reusable model components — single import surface for all halo models.

Usage:
    from models.components import Attention, CodaAttention, MTPHead
    from models.components import ShortConvBlock, MoDAGQABlock
    from models.components import FactorizedEmbedding, FactorizedLMHead
"""

from models.components.attention import Attention, CodaAttention, NoPECodaAttention
from models.components.conv_blocks import (
    ShortConvBlock, GQABlock, MoDAGQABlock, HyPEShortConvBlock,
)
from models.components.embeddings import FactorizedEmbedding, FactorizedLMHead
from models.components.injection import SimpleParcaeInjection
from models.components.loop_utils import HyperloopHC, DepthMemoryCache
from models.components.mtp import MTPHead
from models.components.speculative import (
    DraftHeads, ForecastEmbeddings, concurrent_generate,
)