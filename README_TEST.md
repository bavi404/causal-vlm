# Testing the Framework with test.csv

This guide shows how to test the framework using the `test.csv` file in the root directory.

## Step 1: Convert test.csv to Framework Format

The `test.csv` file is in AudioCaps format. Convert it to the framework's expected format:

```bash
python convert_test_csv.py
```

This will create `data/test_audiocaps/annotations.csv` with the correct format.

## Step 2: Prepare Audio Files (Optional)

The framework needs actual audio files to compute embeddings. You have two options:

### Option A: Use Mock Embeddings (No Audio Files Needed)
Run the test script that uses mock embeddings:

```bash
python test_framework.py
```

This tests:
- Dataset loading
- Interventions (present, masked)
- Fusion strategies (early, late, multimodal)
- Retrieval evaluation

### Option B: Use Real Audio Files
1. Download audio files from YouTube using the `youtube_id` and `start_time` from the CSV
2. Place them in `data/test_audiocaps/audio/` with format: `{youtube_id}_{start_time}.wav`
3. Then run the full evaluation

## Step 3: Run Framework Test

### Quick Test (Mock Embeddings)
```bash
python test_framework.py
```

### Full Test (Requires Audio Files and Model)
```bash
python src/run_baselines.py \
    --dataset audiocaps \
    --fusion early \
    --intervention present \
    --annotations data/test_audiocaps/annotations.csv \
    --data-root data/test_audiocaps
```

## Test CSV Format

The original `test.csv` has:
- `audiocap_id`: Unique identifier
- `youtube_id`: YouTube video ID
- `start_time`: Start time in seconds
- `caption`: Audio caption/description

The converted format has:
- `audio_path`: Path to audio file
- `question`: Caption (used as question)
- `answer`: Caption (used as answer)
- `caption`: Original caption

## Expected Results

The test will:
1. ✅ Load the dataset successfully
2. ✅ Compute embeddings (if audio files available)
3. ✅ Apply interventions (present, masked, swapped)
4. ✅ Evaluate retrieval metrics (R@1, R@5, R@10)
5. ✅ Evaluate QA metrics (if questions/answers available)
6. ✅ Save results to CSV

## Troubleshooting

- **No audio files**: Use `test_framework.py` for mock testing
- **Model not found**: Ensure ImageBind model weights are downloaded
- **File not found errors**: Check that audio paths in CSV match actual file locations
