import torch
import sys
from pathlib import Path

# Add project root (transgtr) to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from models.attention import MultiHeadSelfAttention


def test_attention():
    x = torch.randn(8, 12, 64)

    attention = MultiHeadSelfAttention(
        embed_dim=64,
        num_heads=8,
    )

    y, w = attention(x)

    print("Output Shape :", y.shape)
    print("Attention Shape :", w.shape)


if __name__ == "__main__":
    test_attention()