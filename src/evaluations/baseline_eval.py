"""Baseline evaluation methods for retrieval and QA tasks."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding tensor [N1, D] or [1, D]
        emb2: Second embedding tensor [N2, D] or [1, D]
        
    Returns:
        Cosine similarity matrix [N1, N2] or scalar if both are [1, D]
    """
    # Ensure embeddings are 2D
    if emb1.ndim == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.ndim == 1:
        emb2 = emb2.unsqueeze(0)
    
    # Normalize embeddings (should already be normalized, but ensure)
    emb1_norm = F.normalize(emb1, p=2, dim=-1)
    emb2_norm = F.normalize(emb2, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = torch.mm(emb1_norm, emb2_norm.t())
    return similarity


def compute_retrieval_metrics(
    image_embeddings: List[torch.Tensor],
    audio_embeddings: List[torch.Tensor],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics (Recall@K) using cosine similarity.
    
    Args:
        image_embeddings: List of image embedding tensors, each of shape [1, D]
        audio_embeddings: List of audio embedding tensors, each of shape [1, D]
        k_values: List of K values for Recall@K computation
        
    Returns:
        Dictionary with metrics: {'R@1': ..., 'R@5': ..., 'R@10': ..., 'mean_similarity': ...}
    """
    assert len(image_embeddings) == len(audio_embeddings), \
        f"Number of image ({len(image_embeddings)}) and audio ({len(audio_embeddings)}) embeddings must match"
    
    # Stack embeddings into matrices
    image_matrix = torch.cat(image_embeddings, dim=0)  # [N, D]
    audio_matrix = torch.cat(audio_embeddings, dim=0)  # [N, D]
    
    # Compute similarity matrix [N, N]
    similarity_matrix = cosine_similarity(image_matrix, audio_matrix)
    
    # For each query (image), find ranking of corresponding audio
    # Diagonal should be highest for perfect retrieval
    correct_indices = torch.arange(len(image_embeddings), device=similarity_matrix.device)
    
    # Get similarity scores for correct pairs
    correct_scores = similarity_matrix[correct_indices, correct_indices]
    mean_similarity = correct_scores.mean().item()
    
    # For each image, get ranking of its corresponding audio
    rankings = []
    for i in range(len(image_embeddings)):
        # Get similarities for this image to all audios
        similarities = similarity_matrix[i]
        # Sort in descending order and find rank of correct audio (index i)
        sorted_indices = torch.argsort(similarities, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        rankings.append(rank)
    
    # Compute Recall@K
    metrics = {}
    for k in k_values:
        recall_at_k = sum(1 for rank in rankings if rank <= k) / len(rankings)
        metrics[f'R@{k}'] = recall_at_k
    
    metrics['mean_similarity'] = mean_similarity
    metrics['mean_rank'] = np.mean(rankings)
    
    return metrics


def evaluate_retrieval(
    image_embeddings: List[torch.Tensor],
    audio_embeddings: List[torch.Tensor],
    output_csv: Optional[Union[str, Path]] = None,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Evaluate retrieval performance and optionally log to CSV.
    
    Args:
        image_embeddings: List of image embedding tensors
        audio_embeddings: List of audio embedding tensors
        output_csv: Optional path to CSV file for logging metrics
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with retrieval metrics
    """
    metrics = compute_retrieval_metrics(image_embeddings, audio_embeddings, k_values)
    
    # Log to CSV if requested
    if output_csv:
        log_metrics_to_csv(metrics, output_csv, task='retrieval')
    
    return metrics


def evaluate_qa_simple(
    image_embeddings: List[torch.Tensor],
    audio_embeddings: List[torch.Tensor],
    questions: List[str],
    answer_options: List[List[str]],  # List of answer option strings for each question
    correct_indices: List[int],  # Index of correct answer for each question
    model,
    fuse_method: str = 'mean',
    output_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Simple QA evaluation using embedding similarity.
    
    For each question:
    1. Compute fused embedding: fuse(image_emb, audio_emb)
    2. Compute embeddings for all answer options
    3. predicted_answer = argmax cosine(text_answer_embs, fused_emb)
    
    This reproduces ImageBind paper sanity tests.
    
    Args:
        image_embeddings: List of image embedding tensors [1, D]
        audio_embeddings: List of audio embedding tensors [1, D]
        questions: List of question strings (for reference, not used in computation)
        answer_options: List of lists, each containing answer option strings
        correct_indices: Index of correct answer for each question
        model: Model with get_embeddings method
        fuse_method: How to fuse image and audio ('mean', 'concat', 'sum')
        output_csv: Optional path to CSV file for logging metrics
        
    Returns:
        Dictionary with QA metrics: {'accuracy': ..., 'mean_similarity': ...}
    """
    assert len(image_embeddings) == len(audio_embeddings) == len(questions) == len(answer_options) == len(correct_indices), \
        "Number of images, audios, questions, answer_options, and correct_indices must match"
    
    correct_predictions = 0
    similarities_list = []
    
    for i in tqdm(range(len(image_embeddings)), desc="Evaluating QA"):
        img_emb = image_embeddings[i]  # [1, D]
        audio_emb = audio_embeddings[i]  # [1, D]
        
        # Fuse image and audio embeddings
        if fuse_method == 'mean':
            fused_emb = (img_emb + audio_emb) / 2.0
            # Re-normalize after mean
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)
        elif fuse_method == 'sum':
            fused_emb = img_emb + audio_emb
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)
        elif fuse_method == 'concat':
            fused_emb = torch.cat([img_emb, audio_emb], dim=-1)
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown fuse_method: {fuse_method}")
        
        # Compute embeddings for all answer options
        text_answer_embs = []
        for answer_option in answer_options[i]:
            answer_emb = model.get_embeddings(text=answer_option)
            text_answer_embs.append(answer_emb['text'])
        
        # Stack answer embeddings [num_options, D]
        text_answer_matrix = torch.cat(text_answer_embs, dim=0)
        
        # Compute cosine similarities: [1, num_options]
        similarities = cosine_similarity(fused_emb, text_answer_matrix)
        
        # predicted_answer = argmax cosine(text_answer_embs, fused_emb)
        predicted_idx = similarities.argmax(dim=1).item()
        correct_idx = correct_indices[i]
        
        if predicted_idx == correct_idx:
            correct_predictions += 1
        
        # Store similarity to correct answer
        correct_similarity = similarities[0, correct_idx].item()
        similarities_list.append(correct_similarity)
    
    accuracy = correct_predictions / len(image_embeddings)
    mean_similarity = np.mean(similarities_list)
    
    metrics = {
        'accuracy': accuracy,
        'mean_similarity': mean_similarity,
        'correct_predictions': correct_predictions,
        'total_samples': len(image_embeddings),
    }
    
    # Log to CSV if requested
    if output_csv:
        log_metrics_to_csv(metrics, output_csv, task='qa')
    
    return metrics


def evaluate_qa_multichoice(
    image_embeddings: List[torch.Tensor],
    audio_embeddings: List[torch.Tensor],
    questions: List[str],
    answer_options: List[List[str]],  # List of answer option lists for each question
    correct_indices: List[int],  # Index of correct answer for each question
    model,
    fuse_method: str = 'mean',
    output_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    QA evaluation with multiple choice answers.
    
    For each question, compute fused embedding and find answer option
    with highest cosine similarity. Compare against correct answer index.
    
    Args:
        image_embeddings: List of image embedding tensors [1, D]
        audio_embeddings: List of audio embedding tensors [1, D]
        questions: List of question strings
        answer_options: List of lists, each containing answer option strings
        correct_indices: List of correct answer indices
        model: Model with get_embeddings method
        fuse_method: How to fuse image and audio ('mean', 'concat', 'sum')
        output_csv: Optional path to CSV file for logging metrics
        
    Returns:
        Dictionary with QA metrics
    """
    assert len(image_embeddings) == len(audio_embeddings) == len(questions) == len(answer_options) == len(correct_indices)
    
    correct_predictions = 0
    similarities_list = []
    
    for i in tqdm(range(len(image_embeddings)), desc="Evaluating QA (multi-choice)"):
        img_emb = image_embeddings[i]
        audio_emb = audio_embeddings[i]
        
        # Fuse image and audio embeddings
        if fuse_method == 'mean':
            fused_emb = (img_emb + audio_emb) / 2.0
        elif fuse_method == 'sum':
            fused_emb = img_emb + audio_emb
        elif fuse_method == 'concat':
            fused_emb = torch.cat([img_emb, audio_emb], dim=-1)
            fused_emb = F.normalize(fused_emb, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown fuse_method: {fuse_method}")
        
        # Compute embeddings for all answer options
        answer_option_embs = []
        for answer_option in answer_options[i]:
            answer_emb = model.get_embeddings(text=answer_option)
            answer_option_embs.append(answer_emb['text'])
        
        # Stack answer embeddings [num_options, D]
        answer_matrix = torch.cat(answer_option_embs, dim=0)
        
        # Compute similarities [1, num_options]
        similarities = cosine_similarity(fused_emb, answer_matrix)
        
        # Get predicted answer (argmax)
        predicted_idx = similarities.argmax(dim=1).item()
        correct_idx = correct_indices[i]
        
        if predicted_idx == correct_idx:
            correct_predictions += 1
        
        # Store similarity to correct answer
        correct_similarity = similarities[0, correct_idx].item()
        similarities_list.append(correct_similarity)
    
    accuracy = correct_predictions / len(image_embeddings)
    mean_similarity = np.mean(similarities_list)
    
    metrics = {
        'accuracy': accuracy,
        'mean_similarity': mean_similarity,
        'correct_predictions': correct_predictions,
        'total_samples': len(image_embeddings),
    }
    
    # Log to CSV if requested
    if output_csv:
        log_metrics_to_csv(metrics, output_csv, task='qa_multichoice')
    
    return metrics


def log_metrics_to_csv(
    metrics: Dict[str, float],
    output_path: Union[str, Path],
    task: str = 'evaluation',
    append: bool = False,
):
    """
    Log metrics to CSV file.
    
    Args:
        metrics: Dictionary of metric names to values
        output_path: Path to CSV file
        task: Task name/identifier
        append: Whether to append to existing file or overwrite
    """
    output_path = Path(output_path)
    file_exists = output_path.exists() and append
    
    mode = 'a' if append else 'w'
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow(['task', 'metric', 'value'])
        
        # Write metrics
        for metric_name, metric_value in metrics.items():
            writer.writerow([task, metric_name, metric_value])


def load_metrics_from_csv(csv_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load metrics from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary mapping task names to metric dictionaries
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    
    metrics_by_task = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row['task']
            metric = row['metric']
            value = float(row['value'])
            
            if task not in metrics_by_task:
                metrics_by_task[task] = {}
            metrics_by_task[task][metric] = value
    
    return metrics_by_task

