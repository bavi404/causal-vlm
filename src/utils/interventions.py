"""Intervention functions for causal testing."""

from typing import Callable, Dict
import torch


def audio_present(audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Intervention: Audio is present (no change).
    
    Args:
        audio_emb: Audio embedding tensor
        
    Returns:
        Audio embedding unchanged
    """
    return audio_emb


def audio_masked(audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Intervention: Audio is masked (zeroed out).
    
    Args:
        audio_emb: Audio embedding tensor
        
    Returns:
        Zero tensor with same shape and device as audio_emb
    """
    return torch.zeros_like(audio_emb)


def audio_swapped(audio_emb_A: torch.Tensor, audio_emb_B: torch.Tensor) -> torch.Tensor:
    """
    Intervention: Swap audio embedding A with audio embedding B.
    
    Args:
        audio_emb_A: Original audio embedding (not used in output)
        audio_emb_B: Audio embedding to swap in
        
    Returns:
        audio_emb_B
    """
    return audio_emb_B


def apply_intervention(
    image_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    intervention_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Apply an intervention function to audio embedding and return combined embeddings.
    
    Args:
        image_emb: Image embedding tensor
        audio_emb: Audio embedding tensor
        intervention_fn: Intervention function to apply (e.g., audio_present, audio_masked)
                        or function that takes (audio_emb_A, audio_emb_B) for swapping
        
    Returns:
        Dictionary with keys 'image' and 'audio' containing the embeddings.
        Audio embedding is the result of applying intervention_fn.
    """
    # Apply intervention to audio embedding
    intervened_audio_emb = intervention_fn(audio_emb)
    
    return {
        "image": image_emb,
        "audio": intervened_audio_emb,
    }


def apply_swap_intervention(
    image_emb: torch.Tensor,
    audio_emb_A: torch.Tensor,
    audio_emb_B: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Apply audio swap intervention.
    
    Convenience function for swapping audio embeddings.
    
    Args:
        image_emb: Image embedding tensor
        audio_emb_A: Original audio embedding
        audio_emb_B: Audio embedding to swap in
        
    Returns:
        Dictionary with keys 'image' and 'audio' containing the embeddings.
        Audio embedding is audio_emb_B.
    """
    intervened_audio_emb = audio_swapped(audio_emb_A, audio_emb_B)
    
    return {
        "image": image_emb,
        "audio": intervened_audio_emb,
    }


