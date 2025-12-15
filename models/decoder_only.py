# models/decoder_only.py
import torch
import torch.nn as nn
from typing import Optional
from utils.config import GPTConfig
from .modules.embeddings import TokenAndPositionEmbedding
from .modules.decoder_layer import TransformerDecoderLayer
from .modules.prediction_head import PredictionHead


class GPTModel(nn.Module):
    """
    Decoder-only transformer model for causal language modeling (CLM).
    Architecture:
        Input → Embeddings → Transformer Decoder Stack (no cross-attention) → LM Head

    Note:
        - All decoder layers use causal self-attention
        - No encoder, no cross-attention
        - Weight-tied language modeling head
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer decoder (used as decoder-only)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.ffn_hidden_size,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
                use_cross_attn=False, # decoder only
            )
            for _ in range(config.num_layers)
        ])

        # LM head (weight-tied)
        embedding_weight = self.embeddings.get_input_embeddings().weight
        self.lm_head = PredictionHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            layer_norm_eps=config.layer_norm_eps,
            tie_word_embeddings=True,
            embedding_weight=embedding_weight,
            use_transform=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] — 1 = real token, 0 = padded

        Returns:
            lm_logits: [batch_size, seq_len, vocab_size]
        """
        # Embeddings
        hidden_states = self.embeddings(input_ids)

        # Decoder stack (no encoder)
        for layer in self.decoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None,  # ← no cross-attention
                encoder_attention_mask=None,
            )

        # LM head
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class GPTForPreTraining(nn.Module):
    """
    Wrapper for pretraining: returns language modeling logits.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gpt = GPTModel(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            lm_logits: [batch_size, seq_len, vocab_size]
        """
        return self.gpt(input_ids, attention_mask=attention_mask)