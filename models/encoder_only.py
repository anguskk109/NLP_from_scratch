# models/encoder_only.py
import torch
import torch.nn as nn
from typing import Optional
from utils.config import BertConfig
from .modules.embeddings import TokenAndPositionEmbedding
from .modules.encoder_layer import TransformerEncoderLayer
from .modules.prediction_head import PredictionHead


class BertModel(nn.Module):
    """
    Encoder-only transformer model for masked language modeling (MLM).
    Architecture:
        Input → Embeddings → Transformer Encoder Stack → MLM Head

    Note:
        - No Next Sentence Prediction (NSP)
        - Uses pre-normalization in transformer layers
        - No LayerNorm after embeddings (first encoder layer handles it)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config

        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_seq_len,
            dropout=config.dropout,
        )

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

        embedding_weight = self.embeddings.get_input_embeddings().weight
        self.mlm_head = PredictionHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            layer_norm_eps=config.layer_norm_eps,
            tie_word_embeddings=True,
            embedding_weight=embedding_weight,
            use_transform=True, # for BERT
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
            mlm_logits: [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.embeddings(input_ids)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        mlm_logits = self.mlm_head(hidden_states)
        return mlm_logits


class BertForPreTraining(nn.Module):
    """
    Wrapper for pretraining: returns MLM logits.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            mlm_logits: [batch_size, seq_len, vocab_size]
        """
        return self.bert(input_ids, attention_mask=attention_mask)