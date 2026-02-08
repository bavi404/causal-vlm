"""Run baseline evaluations with interventions and fusion strategies."""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Dict
import torch
import numpy as np

from src.models.imagebind_wrapper import ImageBindWrapper
from src.datasets.music_avqa import MusicAVQADataset, compute_and_save_embeddings, load_embeddings
from src.datasets.avqa import AVQADataset, compute_and_save_embeddings as avqa_compute_embeddings, load_embeddings as avqa_load_embeddings
from src.datasets.audiocaps import AudioCapsDataset, compute_and_save_embeddings as audiocaps_compute_embeddings, load_embeddings as audiocaps_load_embeddings
from src.utils.interventions import audio_present, audio_masked, audio_swapped
from src.utils.fusion import early_fusion, late_fusion, multimodal_fusion
from src.evaluations.baseline_eval import evaluate_retrieval, evaluate_qa_simple


# Dataset configuration
DATASET_CONFIGS = {
    'music-avqa': {
        'class': MusicAVQADataset,
        'compute_fn': compute_and_save_embeddings,
        'load_fn': load_embeddings,
    },
    'avqa': {
        'class': AVQADataset,
        'compute_fn': avqa_compute_embeddings,
        'load_fn': avqa_load_embeddings,
    },
    'audiocaps': {
        'class': AudioCapsDataset,
        'compute_fn': audiocaps_compute_embeddings,
        'load_fn': audiocaps_load_embeddings,
    },
}

# Fusion function mapping
FUSION_FUNCTIONS = {
    'early': early_fusion,
    'late': late_fusion,
    'multimodal': multimodal_fusion,
}

# Intervention function mapping
INTERVENTION_FUNCTIONS = {
    'present': audio_present,
    'masked': audio_masked,
    'swapped': audio_swapped,
}


def load_dataset(dataset_name: str, annotations_file: str, data_root: Optional[str] = None):
    """Load dataset by name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    dataset_class = DATASET_CONFIGS[dataset_name]['class']
    return dataset_class(annotations_file=annotations_file, data_root=data_root)


def get_embeddings_cache_path(dataset_name: str, cache_dir: str = "cache/embeddings") -> Path:
    """Get path for cached embeddings."""
    cache_path = Path(cache_dir) / dataset_name
    return cache_path


def load_or_compute_embeddings(
    dataset,
    model: ImageBindWrapper,
    dataset_name: str,
    cache_dir: str = "cache/embeddings",
    force_recompute: bool = False,
):
    """
    Load embeddings from cache or compute them.
    
    Returns:
        Tuple of (image_embeddings, audio_embeddings) lists
    """
    cache_path = get_embeddings_cache_path(dataset_name, cache_dir)
    image_cache = cache_path / "image_embeddings.pt"
    audio_cache = cache_path / "audio_embeddings.pt"
    
    # Check if cache exists
    if not force_recompute and image_cache.exists() and audio_cache.exists():
        print(f"Loading embeddings from cache: {cache_path}")
        load_fn = DATASET_CONFIGS[dataset_name]['load_fn']
        return load_fn(cache_path)
    
    # Compute embeddings
    print(f"Computing embeddings and saving to cache: {cache_path}")
    compute_fn = DATASET_CONFIGS[dataset_name]['compute_fn']
    image_embs, audio_embs = compute_fn(
        dataset=dataset,
        model=model,
        output_dir=cache_path,
    )
    
    return image_embs, audio_embs


def apply_intervention_to_embeddings(
    image_embeddings: List[torch.Tensor],
    audio_embeddings: List[torch.Tensor],
    intervention_type: str,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Apply intervention to audio embeddings.
    
    Args:
        image_embeddings: List of image embeddings
        audio_embeddings: List of audio embeddings
        intervention_type: Type of intervention ('present', 'masked', 'swapped')
        seed: Random seed for swapping
        
    Returns:
        List of intervened audio embeddings
    """
    if intervention_type == 'present':
        # No change
        return audio_embeddings
    
    elif intervention_type == 'masked':
        # Zero out all audio embeddings
        return [torch.zeros_like(audio_emb) for audio_emb in audio_embeddings]
    
    elif intervention_type == 'swapped':
        # Swap each audio embedding with a random other one
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        intervened_audio = []
        indices = list(range(len(audio_embeddings)))
        
        for i in range(len(audio_embeddings)):
            # Pick a random different index
            swap_idx = random.choice([j for j in indices if j != i])
            intervened_audio.append(audio_embeddings[swap_idx].clone())
        
        return intervened_audio
    
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


def prepare_qa_data(dataset, model: ImageBindWrapper):
    """
    Prepare QA data: extract questions and compute answer embeddings.
    
    Returns:
        Tuple of (questions, answer_options, correct_indices)
    """
    questions = []
    answer_options = []
    correct_indices = []
    
    # Collect unique answers to create answer options
    unique_answers = set()
    for i in range(len(dataset)):
        item = dataset[i]
        answer = item['answer']
        if answer:
            unique_answers.add(answer)
    
    unique_answers = sorted(list(unique_answers))
    print(f"Found {len(unique_answers)} unique answers")
    
    # For each sample, create answer options and find correct index
    for i in range(len(dataset)):
        item = dataset[i]
        questions.append(item['question'])
        
        # Use all unique answers as options (or just a subset for efficiency)
        # In practice, you might want to limit this or use dataset-specific answer sets
        answer_options.append(unique_answers)
        
        # Find correct answer index
        correct_answer = item['answer']
        if correct_answer in unique_answers:
            correct_indices.append(unique_answers.index(correct_answer))
        else:
            # If answer not in options, use first option as placeholder
            correct_indices.append(0)
    
    return questions, answer_options, correct_indices


def run_baseline_evaluation(
    dataset_name: str,
    fusion_type: str,
    intervention: str,
    annotations_file: str,
    data_root: Optional[str] = None,
    cache_dir: str = "cache/embeddings",
    results_dir: str = "results",
    device: Optional[str] = None,
    force_recompute: bool = False,
    swap_seed: int = 42,
):
    """
    Run baseline evaluation with specified fusion and intervention.
    
    Args:
        dataset_name: Name of dataset ('music-avqa', 'avqa', 'audiocaps')
        fusion_type: Fusion strategy ('early', 'late', 'multimodal')
        intervention: Intervention type ('present', 'masked', 'swapped')
        annotations_file: Path to dataset annotations file
        data_root: Root directory for data files
        cache_dir: Directory for caching embeddings
        results_dir: Directory for saving results
        device: Device to run model on
        force_recompute: Force recomputation of embeddings
        swap_seed: Random seed for swap intervention
    """
    print(f"\n{'='*60}")
    print(f"Running Baseline Evaluation")
    print(f"Dataset: {dataset_name}")
    print(f"Fusion: {fusion_type}")
    print(f"Intervention: {intervention}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading ImageBind model...")
    try:
        model = ImageBindWrapper(device=device)
        print(f"Model loaded on device: {model.device}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure ImageBind model weights are available.")
        raise
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, annotations_file, data_root)
        print(f"Dataset loaded: {len(dataset)} samples\n")
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty after loading from {annotations_file}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Load or compute embeddings
    image_embeddings, audio_embeddings = load_or_compute_embeddings(
        dataset=dataset,
        model=model,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )
    
    # Filter out None embeddings
    valid_indices = [
        i for i in range(len(image_embeddings))
        if image_embeddings[i] is not None and audio_embeddings[i] is not None
    ]
    image_embeddings = [image_embeddings[i] for i in valid_indices]
    audio_embeddings = [audio_embeddings[i] for i in valid_indices]
    print(f"Using {len(image_embeddings)} valid samples\n")
    
    # Apply intervention
    print(f"Applying intervention: {intervention}")
    intervened_audio_embeddings = apply_intervention_to_embeddings(
        image_embeddings,
        audio_embeddings,
        intervention_type=intervention,
        seed=swap_seed,
    )
    print("Intervention applied\n")
    
    # For retrieval: cross-modal retrieval doesn't use fusion directly
    # We compare image embeddings to audio embeddings
    # Fusion is applied in QA where we compare fused embeddings to text
    print("Running retrieval evaluation...")
    retrieval_metrics = evaluate_retrieval(
        image_embeddings=image_embeddings,
        audio_embeddings=intervened_audio_embeddings,
        k_values=[1, 5, 10],
    )
    print(f"Retrieval metrics: {retrieval_metrics}\n")
    
    # Run QA evaluation (if dataset has questions/answers)
    qa_metrics = None
    try:
        # Check if dataset has questions and answers
        sample_item = dataset[0]
        if 'question' in sample_item and 'answer' in sample_item and sample_item['answer']:
            print("Running QA evaluation...")
            
            # Prepare QA data
            questions, answer_options, correct_indices = prepare_qa_data(dataset, model)
            
            # Filter to valid indices
            questions = [questions[i] for i in valid_indices]
            answer_options = [answer_options[i] for i in valid_indices]
            correct_indices = [correct_indices[i] for i in valid_indices]
            
            # Map fusion type to fuse_method for evaluate_qa_simple
            fusion_to_fuse_method = {
                'early': 'mean',
                'late': 'concat',
                'multimodal': 'mean',  # For multimodal, we'll need special handling
            }
            fuse_method = fusion_to_fuse_method.get(fusion_type, 'mean')
            
            # For multimodal fusion, we need to pre-fuse embeddings
            if fusion_type == 'multimodal':
                fusion_fn = FUSION_FUNCTIONS[fusion_type]
                # Pre-fuse all embeddings using transformer
                fused_embeddings = []
                for img_emb, audio_emb in zip(image_embeddings, intervened_audio_embeddings):
                    fused = fusion_fn(img_emb, audio_emb)
                    fused_embeddings.append(fused)
                
                # Pass fused embeddings as both image and audio, so mean fusion gives: (fused + fused) / 2 = fused
                qa_metrics = evaluate_qa_simple(
                    image_embeddings=fused_embeddings,
                    audio_embeddings=fused_embeddings,
                    questions=questions,
                    answer_options=answer_options,
                    correct_indices=correct_indices,
                    model=model,
                    fuse_method='mean',  # (fused + fused) / 2 = fused
                )
            else:
                # Run QA evaluation (fusion is done inside evaluate_qa_simple)
                qa_metrics = evaluate_qa_simple(
                    image_embeddings=image_embeddings,
                    audio_embeddings=intervened_audio_embeddings,
                    questions=questions,
                    answer_options=answer_options,
                    correct_indices=correct_indices,
                    model=model,
                    fuse_method=fuse_method,
                )
            print(f"QA metrics: {qa_metrics}\n")
    except Exception as e:
        print(f"QA evaluation skipped: {e}\n")
    
    # Save results
    experiment_name = f"{fusion_type}_{intervention}"
    results_path = Path(results_dir) / dataset_name / f"{experiment_name}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to: {results_path}")
    
    # Combine all metrics
    all_metrics = {
        **{f'retrieval_{k}': v for k, v in retrieval_metrics.items()},
    }
    if qa_metrics:
        all_metrics.update({f'qa_{k}': v for k, v in qa_metrics.items()})
    
    # Add metadata
    all_metrics['dataset'] = dataset_name
    all_metrics['fusion'] = fusion_type
    all_metrics['intervention'] = intervention
    all_metrics['num_samples'] = len(image_embeddings)
    
    # Log to CSV
    from src.evaluations.baseline_eval import log_metrics_to_csv
    try:
        log_metrics_to_csv(
            metrics=all_metrics,
            output_path=results_path,
            task=experiment_name,
            append=False,
        )
    except Exception as e:
        print(f"Warning: Error saving results to CSV: {e}")
        # Try to save as a simple dict instead
        import json
        json_path = results_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved results to JSON instead: {json_path}")
    
    print(f"Results saved to {results_path}")
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations with interventions")
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['music-avqa', 'avqa', 'audiocaps'],
        help='Dataset name'
    )
    parser.add_argument(
        '--fusion',
        type=str,
        required=True,
        choices=['early', 'late', 'multimodal'],
        help='Fusion strategy'
    )
    parser.add_argument(
        '--intervention',
        type=str,
        required=True,
        choices=['present', 'masked', 'swapped'],
        help='Intervention type'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Path to dataset annotations file (JSON or CSV)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Root directory for data files (if paths are relative)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='cache/embeddings',
        help='Directory for caching embeddings'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for saving results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda/cpu, default: auto)'
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation of embeddings (ignore cache)'
    )
    parser.add_argument(
        '--swap-seed',
        type=int,
        default=42,
        help='Random seed for swap intervention'
    )
    
    args = parser.parse_args()
    
    run_baseline_evaluation(
        dataset_name=args.dataset,
        fusion_type=args.fusion,
        intervention=args.intervention,
        annotations_file=args.annotations,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        device=args.device,
        force_recompute=args.force_recompute,
        swap_seed=args.swap_seed,
    )


if __name__ == '__main__':
    main()

