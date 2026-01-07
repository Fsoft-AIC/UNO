import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        
        # Feed-forward network (MLP)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-Attention layer
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feed-forward network (MLP)
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(ff_output)
        # src = self.norm2(src)
        
        return src


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads."

        # Define linear layers for query, key, and value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, embed_dim = x.size()

        # Project the queries, keys, and values
        queries = self.q_proj(x)  # (batch_size, seq_length, embed_dim)
        keys = self.k_proj(x)     # (batch_size, seq_length, embed_dim)
        values = self.v_proj(x)   # (batch_size, seq_length, embed_dim)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attention_output)

        return output

# Example usage
if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embed_dim = 64
    num_heads = 8

    x = torch.randn(batch_size, seq_length, embed_dim)
    self_attention = SelfAttentionLayer(embed_dim, num_heads)
    output, attn_weights = self_attention(x)

    print("Output shape:", output.shape)  # (batch_size, seq_length, embed_dim)
    print("Attention weights shape:", attn_weights.shape)  # (batch_size, num_heads, seq_length, seq_length)



# Example usage
if __name__ == "__main__":
    # Parameters
    d_model = 512
    nhead = 8
    seq_len = 10
    batch_size = 32

    # Initialize layer and input
    encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    src = torch.randn(seq_len, batch_size, d_model)  # (seq_len, batch_size, d_model)

    # Forward pass
    output = encoder_layer(src)
    print(output.shape)  # Should be (seq_len, batch_size, d_model)
