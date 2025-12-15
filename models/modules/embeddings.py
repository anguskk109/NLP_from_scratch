# models/modules/embeddings.py
import torch
import torch.nn as nn
from typing import Optional


class TokenAndPositionEmbedding(nn.Module):
    """
    Token + absolute position embedding layer (NSP-free).
    Used uniformly by BERT (MLM), GPT (CLM), and T5 (span corruption).
    
    Note: No LayerNorm after embedding for BERT — first transformer layer handles normalization
    (using pre-normalization architecture).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings

        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_position_embeddings ({self.max_position_embeddings})")

        token_emb = self.token_emb(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(position_ids)

        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        return embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the token embedding matrix for weight tying in prediction heads."""
        return self.token_emb