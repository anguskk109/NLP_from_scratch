# models/modules/attention.py
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with support for:
    - Padding masks (for variable-length sequences)
    - Causal masks (for autoregressive decoding)
    - Cross-attention (query ≠ key/value)
    """

    def __init__(
            self, 
            hidden_size: int, 
            num_heads: int, 
            dropout: float = 0.1
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self._causal_mask_cache = {}

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # Bias initialized to zero (PyTorch default)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden dim into [batch, seq_len, num_heads, head_dim] and transpose."""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, heads, seq_len, head_dim]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads back to [batch, seq_len, hidden_size]."""
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)
    
    def _causal_mask(self, tgt_len, src_len, device, dtype):
        # cache
        key = (tgt_len, src_len, device, dtype)
        if key in self._causal_mask_cache:
            return self._causal_mask_cache[key]
        mask = torch.full((tgt_len, src_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, src_len]
        self._causal_mask_cache[key] = mask
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, tgt_len, hidden_size]
            key:   [batch_size, src_len, hidden_size]
            value: [batch_size, src_len, hidden_size]
            attn_mask: [batch_size, 1, tgt_len, src_len] (0 = attend, -inf = mask)
            causal: If True, apply causal mask (for decoder self-attention)

        Returns:
            attn_output: [batch_size, tgt_len, hidden_size]
        """
        tgt_len = query.size(1)
        src_len = key.size(1)

        # Project Q, K, V
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        # Split into heads
        q = self._split_heads(q)  # [B, H, tgt_len, D]
        k = self._split_heads(k)  # [B, H, src_len, D]
        v = self._split_heads(v)  # [B, H, src_len, D]

        # Scaled dot-product
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, H, tgt_len, src_len]

        # Attention mask
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Causal mask
        if causal:
            # upper triangle = -inf
            causal_mask = self._causal_mask(tgt_len, src_len, query.device, attn_weights.dtype)
            attn_weights = attn_weights + causal_mask

        # Softmax + dropout
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        attn_output = torch.matmul(attn_probs, v)  # [B, H, tgt_len, D]
        attn_output = self._merge_heads(attn_output)  # [B, tgt_len, hidden_size]
        attn_output = self.out_proj(attn_output)

        return attn_output