# models/modules/decoder_layer.py
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .mlp import MLP


class TransformerDecoderLayer(nn.Module):
    """
    Architecture (pre-normalization):
      1. Self-attention (causal) with residual
      2. Cross-attention (optional) with residual
      3. Feed-forward network with residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention for E+D
        if use_cross_attn:
            self.cross_attn = MultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.cross_attn_dropout = nn.Dropout(dropout)
        else:
            self.cross_attn = None

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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, tgt_len, hidden_size] — decoder input
            attention_mask: [batch_size, tgt_len] — padding mask for decoder input
            if cross-attn:
            encoder_hidden_states: [batch_size, src_len, hidden_size] — from encoder
            encoder_attention_mask: [batch_size, src_len] — padding mask for encoder output

        Returns:
            hidden_states: [batch_size, tgt_len, hidden_size]
        """

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Process self-attention mask (2D → 4D)
        self_attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, seq_len] -> [B, 1, 1, seq_len]
                self_attn_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                self_attn_mask = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}. Expected 2 or 4.")
            
            self_attn_mask = (1.0 - self_attn_mask) * torch.finfo(hidden_states.dtype).min

        self_attn_output = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=self_attn_mask,
            causal=True,  # critical for autoregressive decoding
        )
        self_attn_output = self.self_attn_dropout(self_attn_output)
        hidden_states = residual + self_attn_output

        # Cross-Attention if encoder provided
        if self.cross_attn is not None and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

            # Process encoder attention mask
            cross_attn_mask = None
            if encoder_attention_mask is not None:
                if encoder_attention_mask.dim() == 2:
                    cross_attn_mask = encoder_attention_mask[:, None, None, :]
                elif encoder_attention_mask.dim() == 4:
                    cross_attn_mask = encoder_attention_mask
                else:
                    raise ValueError(f"Unsupported encoder_attention_mask dim: {encoder_attention_mask.dim()}. Expected 2 or 4.")

                cross_attn_mask = (1.0 - cross_attn_mask) * torch.finfo(hidden_states.dtype).min

            cross_attn_output = self.cross_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attn_mask=cross_attn_mask,
                causal=False,  # cross-attention is not causal
            )
            cross_attn_output = self.cross_attn_dropout(cross_attn_output)
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        return hidden_states