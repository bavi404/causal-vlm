"""Quick test script for the framework using test.csv"""

import sys
import warnings
from pathlib import Path
import pandas as pd

# Suppress deprecation warnings from dependencies
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import only what we need for mock testing (avoid loading model)
from src.utils.interventions import audio_present, audio_masked
from src.evaluations.baseline_eval import evaluate_retrieval
from src.utils.fusion import early_fusion, late_fusion, multimodal_fusion

# Lazy import for dataset (only if needed)
try:
    from src.datasets.audiocaps import AudioCapsDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import AudioCapsDataset: {e}")
    DATASET_AVAILABLE = False

def prepare_test_csv(input_csv="test.csv", output_csv="data/test_audiocaps/annotations.csv", max_samples=10):
    """Convert test.csv to format expected by AudioCapsDataset."""
    from pathlib import Path
    import pandas as pd
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Take a small sample for testing
    df_sample = df.head(max_samples).copy()
    
    # Create expected format
    # For AudioCaps, we need: audio_path, question (caption), answer (caption)
    # Since we don't have actual audio files, we'll create placeholder paths
    # In a real scenario, you'd download audio from YouTube using youtube_id
    
    output_data = []
    for idx, row in df_sample.iterrows():
        # Create placeholder audio path (in real use, this would be the actual audio file)
        # Format: audio/{youtube_id}_{start_time}.wav
        audio_path = f"audio/{row['youtube_id']}_{row['start_time']}.wav"
        
        output_data.append({
            'audio_path': audio_path,
            'question': row['caption'],  # Use caption as question
            'answer': row['caption'],    # Use caption as answer (for QA)
            'caption': row['caption'],    # Keep original caption
        })
    
    # Create output directory
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False)
    print(f"Created {output_csv} with {len(output_data)} samples")
    
    return output_path

def test_with_mock_embeddings(save_results=True, output_file="results/test_results.csv"):
    """Test the framework with mock embeddings (no actual model loading)."""
    print("\n" + "="*60)
    print("Testing Framework with Mock Embeddings")
    print("="*60 + "\n")
    
    import torch
    from pathlib import Path
    from src.evaluations.baseline_eval import log_metrics_to_csv
    
    # Create mock embeddings
    num_samples = 10
    embed_dim = 1024
    
    print(f"Creating {num_samples} mock image and audio embeddings...")
    image_embeddings = [torch.randn(1, embed_dim) for _ in range(num_samples)]
    audio_embeddings = [torch.randn(1, embed_dim) for _ in range(num_samples)]
    
    # Normalize
    image_embeddings = [torch.nn.functional.normalize(emb, p=2, dim=-1) for emb in image_embeddings]
    audio_embeddings = [torch.nn.functional.normalize(emb, p=2, dim=-1) for emb in audio_embeddings]
    
    # Test interventions
    print("\nTesting interventions...")
    audio_present_embs = [audio_present(emb) for emb in audio_embeddings]
    audio_masked_embs = [audio_masked(emb) for emb in audio_embeddings]
    
    print(f"Audio present: {len(audio_present_embs)} embeddings")
    print(f"Audio masked: {len(audio_masked_embs)} embeddings")
    print(f"Masked embeddings are zeros: {torch.allclose(audio_masked_embs[0], torch.zeros_like(audio_masked_embs[0]))}")
    
    # Test fusion
    print("\nTesting fusion strategies...")
    early_fused = [early_fusion(img, aud) for img, aud in zip(image_embeddings[:3], audio_embeddings[:3])]
    late_fused = [late_fusion(img, aud) for img, aud in zip(image_embeddings[:3], audio_embeddings[:3])]
    
    print(f"Early fusion shape: {early_fused[0].shape}")
    print(f"Late fusion shape: {late_fused[0].shape}")
    
    # Test retrieval with audio present
    print("\nTesting retrieval evaluation (audio present)...")
    retrieval_metrics_present = evaluate_retrieval(
        image_embeddings=image_embeddings,
        audio_embeddings=audio_embeddings,
        k_values=[1, 5, 10],
    )
    print(f"Retrieval metrics (present): {retrieval_metrics_present}")
    
    # Test retrieval with masked audio
    print("\nTesting retrieval evaluation (audio masked)...")
    retrieval_metrics_masked = evaluate_retrieval(
        image_embeddings=image_embeddings,
        audio_embeddings=audio_masked_embs,
        k_values=[1, 5, 10],
    )
    print(f"Retrieval metrics (masked): {retrieval_metrics_masked}")
    
    # Save results if requested
    if save_results:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine all results
        all_results = {
            # Audio present results
            **{f'present_{k}': v for k, v in retrieval_metrics_present.items()},
            # Audio masked results
            **{f'masked_{k}': v for k, v in retrieval_metrics_masked.items()},
            # Metadata
            'test_type': 'mock_embeddings',
            'num_samples': num_samples,
            'embed_dim': embed_dim,
            'early_fusion_shape': str(early_fused[0].shape),
            'late_fusion_shape': str(late_fused[0].shape),
        }
        
        # Save to CSV
        log_metrics_to_csv(
            metrics=all_results,
            output_path=output_path,
            task='test',
            append=False,
        )
        
        print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Mock Test Complete!")
    print("="*60 + "\n")
    
    return {
        'present': retrieval_metrics_present,
        'masked': retrieval_metrics_masked,
    }

def test_dataset_loading():
    """Test loading the test.csv as a dataset."""
    print("\n" + "="*60)
    print("Testing Dataset Loading")
    print("="*60 + "\n")
    
    if not DATASET_AVAILABLE:
        print("⚠ Skipping dataset loading test (import failed)")
        return
    
    # Prepare CSV
    output_csv = prepare_test_csv(max_samples=10)
    
    # Try to load dataset
    try:
        dataset = AudioCapsDataset(
            annotations_file=str(output_csv),
            data_root=None,
            audio_key="audio_path",
            question_key="question",
            answer_key="answer",
            caption_key="caption",
        )
        print(f"✓ Dataset loaded successfully: {len(dataset)} samples")
        
        # Try to get a sample
        sample = dataset[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        print(f"  Question: {sample['question'][:50]}...")
        print(f"  Audio path: {sample['audio_paths'][0] if sample['audio_paths'] else 'None'}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test framework with mock data")
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save results to CSV file')
    parser.add_argument('--output', type=str, default='results/test_results.csv',
                       help='Output file path for results')
    
    args = parser.parse_args()
    
    print("Framework Test Script")
    print("="*60)
    
    # Test 1: Dataset loading
    test_dataset_loading()
    
    # Test 2: Mock embeddings (no model required)
    results = test_with_mock_embeddings(save_results=args.save_results, output_file=args.output)
    
    print("\nNote: To test with actual model, you need:")
    print("1. ImageBind model weights downloaded")
    print("2. Actual audio files in data/test_audiocaps/audio/")
    print("3. Run: py -m src.run_baselines --dataset audiocaps --fusion early --intervention present --annotations data/test_audiocaps/annotations.csv --data-root data/test_audiocaps")
