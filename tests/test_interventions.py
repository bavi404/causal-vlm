"""Tests for intervention functions."""

import pytest
import torch
import numpy as np

from src.utils.interventions import (
    audio_present,
    audio_masked,
    audio_swapped,
    apply_intervention,
    apply_swap_intervention,
)


class TestAudioPresent:
    """Tests for audio_present intervention."""
    
    def test_audio_present_unchanged(self, audio_embedding):
        """Test that audio_present returns unchanged embedding."""
        result = audio_present(audio_embedding)
        
        assert torch.equal(result, audio_embedding)
        assert result.shape == audio_embedding.shape
        assert result.device == audio_embedding.device
    
    def test_audio_present_preserves_values(self, audio_embedding):
        """Test that audio_present preserves all values."""
        original_values = audio_embedding.clone()
        result = audio_present(audio_embedding)
        
        assert torch.allclose(result, original_values, atol=1e-6)


class TestAudioMasked:
    """Tests for audio_masked intervention."""
    
    def test_audio_masked_all_zeros(self, audio_embedding):
        """Test that audio_masked returns all zeros."""
        result = audio_masked(audio_embedding)
        
        # Check all values are zero
        assert torch.all(result == 0), "Audio masking should produce all zeros"
        assert result.shape == audio_embedding.shape
        assert result.device == audio_embedding.device
    
    def test_audio_masked_shape_preserved(self, audio_embedding):
        """Test that audio_masked preserves shape."""
        original_shape = audio_embedding.shape
        result = audio_masked(audio_embedding)
        
        assert result.shape == original_shape
    
    def test_audio_masked_different_shapes(self):
        """Test audio_masked with different tensor shapes."""
        shapes = [(1, 1024), (5, 1024), (1, 512), (10, 768)]
        
        for shape in shapes:
            audio_emb = torch.randn(*shape)
            result = audio_masked(audio_emb)
            
            assert torch.all(result == 0)
            assert result.shape == shape
    
    def test_audio_masked_device_preserved(self, audio_embedding):
        """Test that audio_masked preserves device."""
        if torch.cuda.is_available():
            audio_emb_cuda = audio_embedding.cuda()
            result = audio_masked(audio_emb_cuda)
            
            assert result.device == audio_emb_cuda.device
            assert torch.all(result == 0)


class TestAudioSwapped:
    """Tests for audio_swapped intervention."""
    
    def test_audio_swapped_returns_b(self, audio_embedding, audio_embedding_2):
        """Test that audio_swapped returns audio_emb_B."""
        result = audio_swapped(audio_embedding, audio_embedding_2)
        
        # Result should match audio_embedding_2 exactly
        assert torch.equal(result, audio_embedding_2)
        assert not torch.equal(result, audio_embedding)  # Should not match A
    
    def test_audio_swapped_shape_preserved(self, audio_embedding, audio_embedding_2):
        """Test that audio_swapped preserves shape of B."""
        result = audio_swapped(audio_embedding, audio_embedding_2)
        
        assert result.shape == audio_embedding_2.shape
    
    def test_audio_swapped_different_shapes(self):
        """Test audio_swapped with different shapes."""
        audio_A = torch.randn(1, 1024)
        audio_B = torch.randn(1, 512)
        
        result = audio_swapped(audio_A, audio_B)
        
        assert torch.equal(result, audio_B)
        assert result.shape == (1, 512)


class TestApplyIntervention:
    """Tests for apply_intervention function."""
    
    def test_apply_intervention_present(self, image_embedding, audio_embedding):
        """Test apply_intervention with audio_present."""
        from src.utils.interventions import audio_present
        
        result = apply_intervention(image_embedding, audio_embedding, audio_present)
        
        assert 'image' in result
        assert 'audio' in result
        assert torch.equal(result['image'], image_embedding)
        assert torch.equal(result['audio'], audio_embedding)
    
    def test_apply_intervention_masked(self, image_embedding, audio_embedding):
        """Test apply_intervention with audio_masked."""
        from src.utils.interventions import audio_masked
        
        result = apply_intervention(image_embedding, audio_embedding, audio_masked)
        
        assert 'image' in result
        assert 'audio' in result
        assert torch.equal(result['image'], image_embedding)
        assert torch.all(result['audio'] == 0)  # Audio should be masked


class TestApplySwapIntervention:
    """Tests for apply_swap_intervention function."""
    
    def test_apply_swap_intervention(self, image_embedding, audio_embedding, audio_embedding_2):
        """Test apply_swap_intervention."""
        result = apply_swap_intervention(image_embedding, audio_embedding, audio_embedding_2)
        
        assert 'image' in result
        assert 'audio' in result
        assert torch.equal(result['image'], image_embedding)
        assert torch.equal(result['audio'], audio_embedding_2)  # Should be swapped to B
        assert not torch.equal(result['audio'], audio_embedding)  # Should not be A


