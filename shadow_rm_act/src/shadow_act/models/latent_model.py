#!/usr/bin/env python3
import torch.nn as nn
from torch.nn import functional as F
import torch

DROPOUT_RATE = 0.1  # Define dropout rate

# Define a Causal Transformer Block
class Causal_Transformer_Block(nn.Module):
    def __init__(self, seq_len, latent_dim, num_head) -> None:
        """
        Initialize the Causal Transformer Block

        Args:
            seq_len (int): Sequence length
            latent_dim (int): Latent dimension
            num_head (int): Number of attention heads
        """
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)  # Layer Normalization
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=DROPOUT_RATE, batch_first=True)  # Multihead Attention
        self.ln_2 = nn.LayerNorm(latent_dim)  # Layer Normalization
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),  # Fully connected layer
            nn.GELU(),  # GELU activation function
            nn.Linear(4 * latent_dim, latent_dim),  # Fully connected layer
            nn.Dropout(DROPOUT_RATE),  # Dropout
        )

        # self.register_buffer("attn_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())  # Register attention mask
    
    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Create upper triangular mask to prevent information leakage
        attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        x = self.ln_1(x)  # Layer Normalization
        x = x + self.attn(x, x, x, attn_mask=attn_mask)[0]  # Add attention output
        x = self.ln_2(x)  # Layer Normalization
        x = x + self.mlp(x)  # Add MLP output
        
        return x

# Use Self-Attention mechanism instead of RNN to model latent space sequences
class Latent_Model_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, latent_dim=256, num_head=8, num_layer=3) -> None:
        """
        Initialize Latent Model Transformer

        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            seq_len (int): Sequence length
            latent_dim (int, optional): Latent dimension, default is 256
            num_head (int, optional): Number of attention heads, default is 8
            num_layer (int, optional): Number of transformer layers, default is 3
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Linear(input_dim, latent_dim)  # Input layer
        self.weight_pos_embed = nn.Embedding(seq_len, latent_dim)  # Positional embedding
        self.attention_blocks = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),  # Dropout
            *[Causal_Transformer_Block(seq_len, latent_dim, num_head) for _ in range(num_layer)],  # Multiple Causal Transformer Blocks
            nn.LayerNorm(latent_dim)  # Layer Normalization
        )
        self.output_layer = nn.Linear(latent_dim, output_dim)  # Output layer
    
    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.input_layer(x)  # Input layer
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))  # Add positional embedding
        x = self.attention_blocks(x)  # Pass through attention blocks
        logits = self.output_layer(x)  # Output layer

        return logits
    
    @torch.no_grad()
    def generate(self, n, temperature=0.1, x=None):
        """
        Generate sequence

        Args:
            n (int): Number of sequences to generate
            temperature (float, optional): Sampling temperature, default is 0.1
            x (torch.Tensor, optional): Initial input tensor, default is None

        Returns:
            torch.Tensor: Generated sequence
        """
        if x is None:
            x = torch.zeros((n, 1, self.input_dim), device=self.weight_pos_embed.weight.device)  # Initialize input
        for i in range(self.seq_len):
            logits = self.forward(x)[:, -1]  # Get output of the last time step
            probs = torch.softmax(logits / temperature, dim=-1)  # Calculate probability distribution
            samples = torch.multinomial(probs, num_samples=1)[..., 0]  # Sample from probability distribution
            samples_one_hot = F.one_hot(samples.long(), num_classes=self.output_dim).float()  # Convert to one-hot encoding
            x = torch.cat([x, samples_one_hot[:, None, :]], dim=1)  # Add new sampled result to input
        
        return x[:, 1:, :]  # Return generated sequence (remove initial zero input)
