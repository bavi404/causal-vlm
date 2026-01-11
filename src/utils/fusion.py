"""Fusion strategies for combining image and audio embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def early_fusion(image_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Early fusion: simple mean of image and audio embeddings.
    
    Args:
        image_emb: Image embedding tensor [B, D] or [1, D]
        audio_emb: Audio embedding tensor [B, D] or [1, D]
        
    Returns:
        Fused embedding tensor [B, D] or [1, D]
    """
    return (image_emb + audio_emb) / 2.0


def late_fusion(image_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Late fusion: concatenation of image and audio embeddings.
    
    Args:
        image_emb: Image embedding tensor [B, D] or [1, D]
        audio_emb: Audio embedding tensor [B, D] or [1, D]
        
    Returns:
        Concatenated embedding tensor [B, 2*D] or [1, 2*D]
    """
    return torch.cat([image_emb, audio_emb], dim=-1)


class MultimodalFusionEncoder(nn.Module):
    """
    Small transformer encoder for multimodal fusion.
    
    Takes concatenated [image; audio] tokens and produces a fused embedding.
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize multimodal fusion encoder.
        
        Args:
            embed_dim: Embedding dimension (should match image/audio embedding dim)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension (defaults to 4 * embed_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        ff_dim = ff_dim or (4 * embed_dim)
        
        # Token embeddings for image and audio
        self.image_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.audio_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Projection layers in case input dims don't match
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.audio_proj = nn.Linear(embed_dim, embed_dim)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 2, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection (CLS token style - use first token)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, image_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: fuse image and audio embeddings.
        
        Args:
            image_emb: Image embedding tensor [B, D] or [1, D]
            audio_emb: Audio embedding tensor [B, D] or [1, D]
            
        Returns:
            Fused embedding tensor [B, D] or [1, D]
        """
        # Ensure 2D tensors [B, D]
        if image_emb.ndim == 1:
            image_emb = image_emb.unsqueeze(0)
        if audio_emb.ndim == 1:
            audio_emb = audio_emb.unsqueeze(0)
        
        batch_size = image_emb.shape[0]
        
        # Project embeddings
        image_proj = self.image_proj(image_emb)  # [B, D]
        audio_proj = self.audio_proj(audio_emb)  # [B, D]
        
        # Add learnable tokens
        image_token = self.image_token.expand(batch_size, -1, -1)  # [B, 1, D]
        audio_token = self.audio_token.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Combine: [image_token + image_emb; audio_token + audio_emb]
        image_combined = image_token + image_proj.unsqueeze(1)  # [B, 1, D]
        audio_combined = audio_token + audio_proj.unsqueeze(1)  # [B, 1, D]
        
        # Stack tokens: [B, 2, D]
        tokens = torch.cat([image_combined, audio_combined], dim=1)
        
        # Add positional encoding
        tokens = tokens + self.pos_embed.expand(batch_size, -1, -1)
        
        # Apply transformer encoder
        fused_tokens = self.transformer(tokens)  # [B, 2, D]
        
        # Use first token (image token) as fused representation
        fused_emb = fused_tokens[:, 0, :]  # [B, D]
        
        # Normalize and project
        fused_emb = self.output_norm(fused_emb)
        fused_emb = self.output_proj(fused_emb)
        
        # L2 normalize
        fused_emb = F.normalize(fused_emb, p=2, dim=-1)
        
        return fused_emb


# Global instance for multimodal fusion (lazy initialization)
_multimodal_fusion_model: Optional[MultimodalFusionEncoder] = None


def multimodal_fusion(
    image_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    model: Optional[MultimodalFusionEncoder] = None,
    embed_dim: int = 1024,
    num_heads: int = 8,
    num_layers: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Multimodal fusion using transformer encoder.
    
    Takes [image; audio] tokens and produces a fused embedding via transformer.
    
    Args:
        image_emb: Image embedding tensor [B, D] or [1, D]
        audio_emb: Audio embedding tensor [B, D] or [1, D]
        model: Optional pre-initialized MultimodalFusionEncoder model
        embed_dim: Embedding dimension (used if model is None)
        num_heads: Number of attention heads (used if model is None)
        num_layers: Number of transformer layers (used if model is None)
        device: Device to run model on (used if model is None)
        
    Returns:
        Fused embedding tensor [B, D] or [1, D]
    """
    global _multimodal_fusion_model
    
    # Use provided model or create/use global instance
    if model is not None:
        fusion_model = model
    else:
        if _multimodal_fusion_model is None:
            # Infer embed_dim from input if not provided
            if image_emb.ndim >= 2:
                inferred_dim = image_emb.shape[-1]
            else:
                inferred_dim = image_emb.shape[0]
            
            embed_dim = embed_dim if embed_dim != 1024 else inferred_dim
            
            _multimodal_fusion_model = MultimodalFusionEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            
            if device is not None:
                _multimodal_fusion_model = _multimodal_fusion_model.to(device)
            elif image_emb.is_cuda:
                _multimodal_fusion_model = _multimodal_fusion_model.to(image_emb.device)
        
        fusion_model = _multimodal_fusion_model
    
    # Ensure model is on same device as inputs
    if image_emb.is_cuda and not next(fusion_model.parameters()).is_cuda:
        fusion_model = fusion_model.to(image_emb.device)
    
    fusion_model.eval()
    with torch.no_grad():
        fused_emb = fusion_model(image_emb, audio_emb)
    
    return fused_emb


# Export all fusion strategies
__all__ = [
    'early_fusion',
    'late_fusion',
    'multimodal_fusion',
    'MultimodalFusionEncoder',
]


