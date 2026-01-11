"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def embedding_dim():
    """Standard embedding dimension for tests."""
    return 1024


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 1


@pytest.fixture
def image_embedding(embedding_dim, batch_size):
    """Create a sample image embedding."""
    return torch.randn(batch_size, embedding_dim)


@pytest.fixture
def audio_embedding(embedding_dim, batch_size):
    """Create a sample audio embedding."""
    return torch.randn(batch_size, embedding_dim)


@pytest.fixture
def audio_embedding_2(embedding_dim, batch_size):
    """Create a second sample audio embedding for swapping tests."""
    return torch.randn(batch_size, embedding_dim)


@pytest.fixture
def multiple_embeddings(embedding_dim):
    """Create multiple embeddings for batch tests."""
    num_samples = 10
    image_embs = torch.randn(num_samples, embedding_dim)
    audio_embs = torch.randn(num_samples, embedding_dim)
    return image_embs, audio_embs


@pytest.fixture
def device():
    """Device for tests."""
    return torch.device('cpu')


