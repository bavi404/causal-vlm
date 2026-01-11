"""Tests for fusion functions."""

import pytest
import torch
import numpy as np

from src.utils.fusion import early_fusion, late_fusion, multimodal_fusion, MultimodalFusionEncoder


class TestEarlyFusion:
    """Tests for early_fusion function."""
    
    def test_early_fusion_shape(self, image_embedding, audio_embedding):
        """Test that early_fusion preserves embedding dimension."""
        result = early_fusion(image_embedding, audio_embedding)
        
        # Should have same dimension as input
        assert result.shape == image_embedding.shape
        assert result.shape == audio_embedding.shape
    
    def test_early_fusion_values(self, image_embedding, audio_embedding):
        """Test that early_fusion computes mean correctly."""
        result = early_fusion(image_embedding, audio_embedding)
        expected = (image_embedding + audio_embedding) / 2.0
        
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_early_fusion_batch(self):
        """Test early_fusion with batch dimension."""
        batch_size = 5
        embed_dim = 1024
        image_emb = torch.randn(batch_size, embed_dim)
        audio_emb = torch.randn(batch_size, embed_dim)
        
        result = early_fusion(image_emb, audio_emb)
        
        assert result.shape == (batch_size, embed_dim)
        expected = (image_emb + audio_emb) / 2.0
        assert torch.allclose(result, expected, atol=1e-6)


class TestLateFusion:
    """Tests for late_fusion function."""
    
    def test_late_fusion_shape(self, image_embedding, audio_embedding):
        """Test that late_fusion doubles embedding dimension."""
        result = late_fusion(image_embedding, audio_embedding)
        
        embed_dim = image_embedding.shape[-1]
        expected_shape = (*image_embedding.shape[:-1], embed_dim * 2)
        
        assert result.shape == expected_shape
    
    def test_late_fusion_concatenation(self, image_embedding, audio_embedding):
        """Test that late_fusion concatenates correctly."""
        result = late_fusion(image_embedding, audio_embedding)
        
        # Check first half is image, second half is audio
        embed_dim = image_embedding.shape[-1]
        image_part = result[..., :embed_dim]
        audio_part = result[..., embed_dim:]
        
        assert torch.allclose(image_part, image_embedding, atol=1e-6)
        assert torch.allclose(audio_part, audio_embedding, atol=1e-6)
    
    def test_late_fusion_batch(self):
        """Test late_fusion with batch dimension."""
        batch_size = 5
        embed_dim = 1024
        image_emb = torch.randn(batch_size, embed_dim)
        audio_emb = torch.randn(batch_size, embed_dim)
        
        result = late_fusion(image_emb, audio_emb)
        
        assert result.shape == (batch_size, embed_dim * 2)


class TestMultimodalFusion:
    """Tests for multimodal_fusion function."""
    
    def test_multimodal_fusion_shape(self, image_embedding, audio_embedding):
        """Test that multimodal_fusion preserves embedding dimension."""
        result = multimodal_fusion(image_embedding, audio_embedding)
        
        # Should have same dimension as input
        assert result.shape == image_embedding.shape
        assert result.shape == audio_embedding.shape
    
    def test_multimodal_fusion_batch(self):
        """Test multimodal_fusion with batch dimension."""
        batch_size = 3
        embed_dim = 1024
        image_emb = torch.randn(batch_size, embed_dim)
        audio_emb = torch.randn(batch_size, embed_dim)
        
        result = multimodal_fusion(image_emb, audio_emb)
        
        assert result.shape == (batch_size, embed_dim)
    
    def test_multimodal_fusion_normalized(self, image_embedding, audio_embedding):
        """Test that multimodal_fusion output is normalized."""
        result = multimodal_fusion(image_embedding, audio_embedding)
        
        # Check L2 norm is approximately 1
        norm = torch.norm(result, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    
    def test_multimodal_fusion_custom_model(self, image_embedding, audio_embedding):
        """Test multimodal_fusion with custom model."""
        embed_dim = image_embedding.shape[-1]
        model = MultimodalFusionEncoder(embed_dim=embed_dim, num_layers=2)
        
        result = multimodal_fusion(image_embedding, audio_embedding, model=model)
        
        assert result.shape == image_embedding.shape


class TestFusionShapes:
    """Tests for fusion shape correctness across all methods."""
    
    @pytest.mark.parametrize("fusion_fn,expected_dim_mult", [
        (early_fusion, 1),
        (late_fusion, 2),
        (multimodal_fusion, 1),
    ])
    def test_fusion_shapes_1d(self, fusion_fn, expected_dim_mult, embedding_dim):
        """Test fusion shapes with 1D tensors."""
        image_emb = torch.randn(embedding_dim)
        audio_emb = torch.randn(embedding_dim)
        
        result = fusion_fn(image_emb, audio_emb)
        
        expected_dim = embedding_dim * expected_dim_mult
        assert result.shape == (expected_dim,)
    
    @pytest.mark.parametrize("fusion_fn,expected_dim_mult", [
        (early_fusion, 1),
        (late_fusion, 2),
        (multimodal_fusion, 1),
    ])
    def test_fusion_shapes_2d(self, fusion_fn, expected_dim_mult, embedding_dim):
        """Test fusion shapes with 2D tensors."""
        batch_size = 1
        image_emb = torch.randn(batch_size, embedding_dim)
        audio_emb = torch.randn(batch_size, embedding_dim)
        
        result = fusion_fn(image_emb, audio_emb)
        
        expected_dim = embedding_dim * expected_dim_mult
        assert result.shape == (batch_size, expected_dim)
    
    @pytest.mark.parametrize("fusion_fn,expected_dim_mult", [
        (early_fusion, 1),
        (late_fusion, 2),
        (multimodal_fusion, 1),
    ])
    def test_fusion_shapes_batch(self, fusion_fn, expected_dim_mult, embedding_dim):
        """Test fusion shapes with batch dimension."""
        batch_size = 10
        image_emb = torch.randn(batch_size, embedding_dim)
        audio_emb = torch.randn(batch_size, embedding_dim)
        
        result = fusion_fn(image_emb, audio_emb)
        
        expected_dim = embedding_dim * expected_dim_mult
        assert result.shape == (batch_size, expected_dim)


