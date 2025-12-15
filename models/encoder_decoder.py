# models/encoder_decoder.py
import torch
import torch.nn as nn
from typing import Optional
from utils.config import T5Config
from .modules.embeddings import TokenAndPositionEmbedding
from .modules.encoder_layer import TransformerEncoderLayer
from .modules.decoder_layer import TransformerDecoderLayer
from .modules.prediction_head import PredictionHead


class T5Model(nn.Module):
    """
    Encoder-decoder transformer model for span corruption pretraining (T5-style).
    Architecture:
        Encoder:   Input → Embeddings → Transformer Encoder Stack
        Decoder:   Target → Embeddings → Transformer Decoder Stack (with cross-attention to encoder)
        Output:    Decoder hidden states → LM Head (weight-tied)
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

        # Shared embedding layer for encoder and decoder
        self.shared_embeddings = TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_seq_len,
            dropout=config.dropout,
        )

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.ffn_hidden_size,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.ffn_hidden_size,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
                use_cross_attn=True # with cross attn
            )
            for _ in range(config.num_layers)
        ])

        # Decoder LM head (weight-tied)
        embedding_weight = self.shared_embeddings.get_input_embeddings().weight
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
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, src_len] — corrupted input
            decoder_input_ids: [batch_size, tgt_len] — target span tokens
            encoder_attention_mask: [batch_size, src_len] — padding mask for encoder input
            decoder_attention_mask: [batch_size, tgt_len] — padding mask for decoder input

        Returns:
            lm_logits: [batch_size, tgt_len, vocab_size]
        """

        encoder_hidden_states = self.shared_embeddings(input_ids)
        for layer in self.encoder_layers:
            encoder_hidden_states = layer(
                hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask
            )

        decoder_hidden_states = self.shared_embeddings(decoder_input_ids)
        for layer in self.decoder_layers:
            decoder_hidden_states = layer(
                hidden_states=decoder_hidden_states,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        lm_logits = self.lm_head(decoder_hidden_states)
        return lm_logits


class T5ForPreTraining(nn.Module):
    """
    Wrapper for T5-style span corruption pretraining.
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.t5 = T5Model(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            lm_logits: [batch_size, tgt_len, vocab_size]
        """
        return self.t5(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )