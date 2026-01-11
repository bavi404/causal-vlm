# Quick Start Guide

## Before Your Presentation

### 1. Verify Everything Works

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Check project structure
ls -la src/
```

### 2. Prepare Sample Data (if needed)

If you don't have real data, create minimal test files:

```bash
# Create sample annotation files
mkdir -p data/music-avqa data/avqa data/audiocaps

# Example JSON format (music-avqa/annotations.json):
cat > data/music-avqa/annotations.json << EOF
[
  {
    "image_path": "images/sample1.jpg",
    "audio_path": "audio/sample1.wav",
    "question": "What instrument is playing?",
    "answer": "piano"
  }
]
EOF
```

### 3. Run a Quick Test

```bash
# Run one experiment to verify
python src/run_baselines.py \
    --dataset music-avqa \
    --fusion early \
    --intervention present \
    --annotations data/music-avqa/annotations.json \
    --data-root data/music-avqa
```

### 4. Generate Results Table

```bash
python src/evaluations/make_results_table.py
```

---

## Presentation Checklist

- [ ] README.md reviewed
- [ ] Project structure understood
- [ ] Key code examples ready
- [ ] Results table generated (or sample ready)
- [ ] Tests pass
- [ ] Demo script prepared
- [ ] Questions anticipated

---

## One-Liner Summary

**"A framework for causal analysis of audio-visual models using interventions (mask/swap audio) to quantify audio's contribution across 3 datasets and 3 fusion strategies."**

---

## If Something Breaks

1. **Import errors**: Check `requirements.txt` installed
2. **ImageBind not found**: Verify `src/models/imagebind/` exists
3. **Dataset errors**: Check annotation file format
4. **CUDA errors**: Script auto-falls back to CPU

---

## Emergency Backup

If live demo fails, have ready:
- Screenshots of results
- Pre-generated results table PNG
- Code snippets in slides
- Test output screenshots

Good luck! 🎯


