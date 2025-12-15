# models/modules/encoder_layer.py
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .mlp import MLP


class TransformerEncoderLayer(nn.Module):
    """
    Architecture -- pre-normalization:
        x → LayerNorm → Self-Attention → Dropout → Add (residual) →
        → LayerNorm → FFN → Dropout → Add (residual)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn_dropout = nn.Dropout(dropout)


        self.ffn = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Process attention_mask to additive 4D form
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, seq_len] -> [B, 1, 1, seq_len]
                attn_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                attn_mask = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}. Expected 2 or 4.")

            # Convert to additive mask: 1 → 0, 0 → -inf
            # so that after SoftMax: 0 → 0, 1 → -inf
            attn_mask = (1.0 - attn_mask) * torch.finfo(hidden_states.dtype).min

        attn_output = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=attn_mask,
            causal=False,  # Encoder is bidirectional
        )
        attn_output = self.self_attn_dropout(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states