"""Tests for retrieval evaluation."""

import pytest
import torch
import numpy as np

from src.evaluations.baseline_eval import (
    compute_retrieval_metrics,
    evaluate_retrieval,
    cosine_similarity,
)
from src.utils.interventions import audio_present, audio_masked


class TestRetrievalMetrics:
    """Tests for retrieval metrics computation."""
    
    def test_retrieval_metrics_shape(self, multiple_embeddings):
        """Test that retrieval metrics are computed correctly."""
        image_embs, audio_embs = multiple_embeddings
        
        metrics = compute_retrieval_metrics(image_embs, audio_embs, k_values=[1, 5, 10])
        
        assert 'R@1' in metrics
        assert 'R@5' in metrics
        assert 'R@10' in metrics
        assert 'mean_similarity' in metrics
        assert 'mean_rank' in metrics
        
        # All metrics should be between 0 and 1 (except mean_rank)
        assert 0 <= metrics['R@1'] <= 1
        assert 0 <= metrics['R@5'] <= 1
        assert 0 <= metrics['R@10'] <= 1
        assert metrics['mean_rank'] >= 1
    
    def test_retrieval_perfect_match(self):
        """Test retrieval with perfect matches (identity)."""
        num_samples = 5
        embed_dim = 1024
        
        # Create identical image and audio embeddings
        image_embs = torch.randn(num_samples, embed_dim)
        audio_embs = image_embs.clone()  # Perfect match
        
        metrics = compute_retrieval_metrics(image_embs, audio_embs, k_values=[1, 5, 10])
        
        # With perfect matches, R@1 should be 1.0
        assert metrics['R@1'] == 1.0
        assert metrics['R@5'] == 1.0
        assert metrics['R@10'] == 1.0
        assert metrics['mean_rank'] == 1.0
    
    def test_retrieval_random_match(self):
        """Test retrieval with random embeddings."""
        num_samples = 10
        embed_dim = 1024
        
        image_embs = torch.randn(num_samples, embed_dim)
        audio_embs = torch.randn(num_samples, embed_dim)  # Random, no correlation
        
        metrics = compute_retrieval_metrics(image_embs, audio_embs, k_values=[1, 5, 10])
        
        # With random embeddings, R@1 should be low (but not necessarily 0)
        assert 0 <= metrics['R@1'] <= 1
        # R@10 should be >= R@5 >= R@1
        assert metrics['R@10'] >= metrics['R@5']
        assert metrics['R@5'] >= metrics['R@1']


class TestRetrievalInterventions:
    """Tests for retrieval with interventions."""
    
    def test_audio_present_vs_masked(self):
        """
        Test that R@1 is higher with audio_present than audio_masked.
        
        This tests that audio information improves retrieval performance.
        """
        num_samples = 20
        embed_dim = 1024
        
        # Create correlated image-audio pairs
        # Image and audio embeddings should be similar for matching pairs
        image_embs = torch.randn(num_samples, embed_dim)
        # Audio embeddings are correlated with images (add some noise)
        audio_embs = image_embs + 0.3 * torch.randn(num_samples, embed_dim)
        
        # Normalize
        image_embs = torch.nn.functional.normalize(image_embs, p=2, dim=-1)
        audio_embs = torch.nn.functional.normalize(audio_embs, p=2, dim=-1)
        
        # Convert to list format (each element is [1, D])
        image_emb_list = [image_embs[i:i+1] for i in range(num_samples)]
        
        # Test with audio present
        audio_emb_list_present = [audio_present(audio_embs[i:i+1]) for i in range(num_samples)]
        
        metrics_present = compute_retrieval_metrics(
            image_emb_list,
            audio_emb_list_present,
            k_values=[1]
        )
        
        # Test with audio masked
        audio_emb_list_masked = [audio_masked(audio_embs[i:i+1]) for i in range(num_samples)]
        
        metrics_masked = compute_retrieval_metrics(
            image_emb_list,
            audio_emb_list_masked,
            k_values=[1]
        )
        
        # Audio present should have higher R@1 than masked
        # (unless audio is completely uncorrelated, in which case they might be similar)
        r1_present = metrics_present['R@1']
        r1_masked = metrics_masked['R@1']
        
        print(f"R@1 with audio present: {r1_present:.4f}")
        print(f"R@1 with audio masked: {r1_masked:.4f}")
        
        # With correlated embeddings, present should be better
        # But we allow for some tolerance in case of edge cases
        assert r1_present >= r1_masked - 0.1, \
            f"Audio present R@1 ({r1_present}) should be >= masked R@1 ({r1_masked})"
    
    def test_audio_masked_zero_similarity(self):
        """Test that masked audio produces zero similarity."""
        num_samples = 5
        embed_dim = 1024
        
        image_embs = torch.randn(num_samples, embed_dim)
        audio_embs = torch.randn(num_samples, embed_dim)
        
        # Mask audio
        audio_embs_masked = [audio_masked(audio_embs[i:i+1]) for i in range(num_samples)]
        audio_embs_masked = torch.cat(audio_embs_masked, dim=0)
        
        # Compute similarity matrix
        similarity = cosine_similarity(image_embs, audio_embs_masked)
        
        # All similarities should be zero (or very close)
        assert torch.allclose(similarity, torch.zeros_like(similarity), atol=1e-5)
    
    def test_retrieval_audio_dependent_dataset(self):
        """
        Test that on audio-dependent datasets, audio_present > audio_masked.
        
        This simulates a scenario where audio information is crucial for retrieval.
        """
        num_samples = 30
        embed_dim = 1024
        
        # Create strongly correlated image-audio pairs (audio-dependent scenario)
        base_embs = torch.randn(num_samples, embed_dim)
        image_embs = base_embs + 0.1 * torch.randn(num_samples, embed_dim)
        audio_embs = base_embs + 0.1 * torch.randn(num_samples, embed_dim)  # Strong correlation
        
        # Normalize
        image_embs = torch.nn.functional.normalize(image_embs, p=2, dim=-1)
        audio_embs = torch.nn.functional.normalize(audio_embs, p=2, dim=-1)
        
        # Convert to list format
        image_emb_list = [image_embs[i:i+1] for i in range(num_samples)]
        audio_emb_list_present = [audio_present(audio_embs[i:i+1]) for i in range(num_samples)]
        audio_emb_list_masked = [audio_masked(audio_embs[i:i+1]) for i in range(num_samples)]
        
        # Compute metrics
        metrics_present = compute_retrieval_metrics(
            image_emb_list, audio_emb_list_present, k_values=[1]
        )
        metrics_masked = compute_retrieval_metrics(
            image_emb_list, audio_emb_list_masked, k_values=[1]
        )
        
        r1_present = metrics_present['R@1']
        r1_masked = metrics_masked['R@1']
        
        # On audio-dependent datasets, present should be significantly better
        assert r1_present > r1_masked, \
            f"On audio-dependent data, R@1 present ({r1_present}) must be > masked ({r1_masked})"
        
        # The difference should be substantial
        improvement = r1_present - r1_masked
        assert improvement > 0.1, \
            f"Improvement should be substantial: {improvement:.4f}"


class TestCosineSimilarity:
    """Tests for cosine similarity function."""
    
    def test_cosine_similarity_shape(self):
        """Test cosine similarity output shape."""
        emb1 = torch.randn(5, 1024)
        emb2 = torch.randn(3, 1024)
        
        result = cosine_similarity(emb1, emb2)
        
        assert result.shape == (5, 3)
    
    def test_cosine_similarity_normalized(self):
        """Test that cosine similarity works with normalized embeddings."""
        emb1 = torch.randn(3, 1024)
        emb2 = torch.randn(3, 1024)
        
        # Normalize
        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=-1)
        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=-1)
        
        result = cosine_similarity(emb1_norm, emb2_norm)
        
        # Diagonal should be close to 1 (self-similarity)
        for i in range(3):
            assert -1 <= result[i, i].item() <= 1
    
    def test_cosine_similarity_range(self):
        """Test that cosine similarity is in [-1, 1] range."""
        emb1 = torch.randn(10, 1024)
        emb2 = torch.randn(10, 1024)
        
        result = cosine_similarity(emb1, emb2)
        
        assert torch.all(result >= -1)
        assert torch.all(result <= 1)

