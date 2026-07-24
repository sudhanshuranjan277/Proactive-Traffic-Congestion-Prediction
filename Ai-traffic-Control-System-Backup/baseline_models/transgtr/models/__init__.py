"""
Models Package

This package contains all model components used in the
TransGTR baseline implementation.

Modules
-------
- Embedding
- Attention
- Transformer Block
- TSFormer
- Graph Structure Generator
- GraphWaveNet
- Knowledge Distillation
- Prediction Head
"""

from .embedding import DataEmbedding

__all__ = [
    "DataEmbedding",
]