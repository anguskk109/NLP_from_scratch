# models/modules/prediction_head.py
import torch
import torch.nn as nn
from typing import Optional


class PredictionHead(nn.Module):
    """  
    Extra transform for BERT:
          → Linear (hidden → hidden) 
          → GELU 
          → LayerNorm

    Simple heads for Decoders.

    Weight tying: 
        the final linear layer's weight matrix is shared with the input token embedding matrix.
        - BERT: MLM head (tied)
        - GPT: LM head (tied)
        - T5: Decoder LM head (tied)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        layer_norm_eps: float = 1e-12,
        tie_word_embeddings: bool = True,
        embedding_weight: Optional[nn.Parameter] = None,
        use_transform: bool = False,
    ):
        super().__init__()
        self.tie_word_embeddings = tie_word_embeddings

        # Transform layer for BERT
        if use_transform:
            self.transform = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            )
            self._init_weights()
        # for the rest:
        else:
            self.transform = nn.Identity()

        # Output projection
        bias = not tie_word_embeddings
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=bias)

        if tie_word_embeddings:
            assert embedding_weight.shape == (vocab_size, hidden_size)
            assert embedding_weight.device == self.decoder.weight.device, "Device mismatch in weight tying!"
            if embedding_weight is None:
                raise ValueError(
                    "tie_word_embeddings=True but embedding_weight is None. "
                    "Pass the token embedding weight from the model's embedding layer."
                )
            # Tie weights: output projection = input embedding
            self.decoder.weight = embedding_weight

        

    def _init_weights(self):
        nn.init.xavier_uniform_(self.transform[0].weight)
        # decoder.bias is already zero

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        return logits