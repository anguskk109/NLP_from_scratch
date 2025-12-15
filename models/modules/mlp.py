# models/modules/mlp.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Sequential: hidden_states → Linear → GELU → Dropout → Linear → Dropout
    
    Note:
    intermediate_size: Dimension of the inner layer, typically 4x hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")  # Matches BERT/GPT

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        # Biases are zero by default in PyTorch

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states