# Causal VLM: Audio-Visual Causal Analysis

A comprehensive framework for evaluating causal relationships in audio-visual multimodal models using ImageBind.

## Project Structure

```
causal-vlm/
├── src/
│   ├── models/
│   │   ├── imagebind/          # Meta's ImageBind (cloned)
│   │   └── imagebind_wrapper.py # Wrapper for easy embedding extraction
│   ├── datasets/
│   │   ├── music_avqa.py       # Music-AVQA dataset loader
│   │   ├── avqa.py             # AVQA dataset loader
│   │   └── audiocaps.py        # AudioCaps dataset loader
│   ├── evaluations/
│   │   ├── baseline_eval.py    # Retrieval & QA evaluation
│   │   └── make_results_table.py # Results aggregation
│   ├── utils/
│   │   ├── interventions.py   # Causal interventions (mask, swap)
│   │   └── fusion.py           # Fusion strategies (early, late, multimodal)
│   └── run_baselines.py        # Main evaluation script
├── tests/                       # Comprehensive test suite
├── run_all.sh                  # Run all experiments (Bash)
├── run_all.ps1                 # Run all experiments (PowerShell)
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your datasets:
```
data/
├── music-avqa/
│   ├── annotations.json
│   ├── images/
│   └── audio/
├── avqa/
│   ├── annotations.csv
│   ├── images/
│   └── audio/
└── audiocaps/
    ├── annotations.json
    ├── images/
    └── audio/
```

### 3. Run Experiments

**Run all experiments:**
```bash
# Linux/Mac/Git Bash
bash run_all.sh

# Windows PowerShell
.\run_all.ps1
```

**Run single experiment:**
```bash
python src/run_baselines.py \
    --dataset music-avqa \
    --fusion early \
    --intervention present \
    --annotations data/music-avqa/annotations.json \
    --data-root data/music-avqa
```

### 4. Generate Results Table

```bash
python src/evaluations/make_results_table.py \
    --results-dir results \
    --output-dir results/tables
```

## Key Features

### 🔬 Causal Interventions
- **Audio Present**: Baseline (no intervention)
- **Audio Masked**: Zero out audio embeddings
- **Audio Swapped**: Swap audio embeddings between samples

### 🔀 Fusion Strategies
- **Early Fusion**: Mean of image + audio embeddings
- **Late Fusion**: Concatenation of embeddings
- **Multimodal Fusion**: Transformer-based fusion

### 📊 Evaluation Metrics
- **Retrieval**: R@1, R@5, R@10
- **QA**: Accuracy, mean similarity

### 📈 Results
- Automatic CSV logging
- Aggregated results tables (Markdown, CSV, PNG)
- Cached embeddings for fast re-evaluation

## Experiments

The framework runs **27 experiments** (3 datasets × 3 interventions × 3 fusion types):

| Dataset | Intervention | Fusion |
|---------|--------------|--------|
| MUSIC-AVQA | present/masked/swapped | early/late/multimodal |
| AVQA | present/masked/swapped | early/late/multimodal |
| Audiocaps | present/masked/swapped | early/late/multimodal |

## Results Location

- **Individual Results**: `results/{dataset}/{fusion}_{intervention}.csv`
- **Aggregated Table**: `results/tables/results_table.{md,csv,png}`
- **Cached Embeddings**: `cache/embeddings/{dataset}/`

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Tests cover:
- ✅ Audio masking produces all zeros
- ✅ Audio swapping matches source
- ✅ Fusion shapes are correct
- ✅ Retrieval R@1 increases with audio present vs masked

## Citation

If you use this framework, please cite:
- ImageBind: [Paper](https://arxiv.org/abs/2305.05665)
- This framework (if applicable)

## License

See individual component licenses (ImageBind uses CC-BY-NC 4.0).


