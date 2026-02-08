"""Convert test.csv to format expected by the framework."""

import pandas as pd
from pathlib import Path

def convert_test_csv(input_csv="test.csv", output_csv="data/test_audiocaps/annotations.csv", max_samples=None):
    """
    Convert test.csv (AudioCaps format) to framework format.
    
    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        max_samples: Maximum number of samples to include (None for all)
    """
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head(3))
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
        print(f"\nUsing first {max_samples} samples for testing")
    
    # Create expected format for AudioCapsDataset
    # The dataset expects: audio_path, question, answer (optional), image_path (optional)
    # For AudioCaps, we use caption as both question and answer
    
    output_data = []
    for idx, row in df.iterrows():
        # Create audio path
        # In real scenario, you'd have actual audio files
        # Format: audio/{youtube_id}_{start_time}.wav or similar
        audio_filename = f"{row['youtube_id']}_{row['start_time']}.wav"
        audio_path = f"audio/{audio_filename}"
        
        output_data.append({
            'audio_path': audio_path,
            'question': row['caption'],  # Use caption as question
            'answer': row['caption'],    # Use caption as answer (for QA evaluation)
            'caption': row['caption'],    # Keep original caption
            # Optional: keep original metadata
            'audiocap_id': row['audiocap_id'],
            'youtube_id': row['youtube_id'],
            'start_time': row['start_time'],
        })
    
    # Create output directory
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Created {output_csv}")
    print(f"  Samples: {len(output_data)}")
    print(f"  Columns: {list(output_df.columns)}")
    print(f"\nFirst sample:")
    print(output_df.iloc[0].to_dict())
    
    print(f"\nNote: Audio files should be placed in: {output_path.parent / 'audio'}")
    print(f"      Expected format: {audio_filename}")
    
    return output_path

if __name__ == "__main__":
    # Convert with a small sample for testing
    convert_test_csv(max_samples=50)
    
    print("\n" + "="*60)
    print("To test the framework, run:")
    print("="*60)
    print("\npython src/run_baselines.py \\")
    print("    --dataset audiocaps \\")
    print("    --fusion early \\")
    print("    --intervention present \\")
    print("    --annotations data/test_audiocaps/annotations.csv \\")
    print("    --data-root data/test_audiocaps")
    print("\nOr use the test script:")
    print("python test_framework.py")
