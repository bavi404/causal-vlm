# PowerShell script to run all baseline experiments across datasets, interventions, and fusion types

$ErrorActionPreference = "Continue"

# Default paths (can be overridden with environment variables)
$ANNOTATIONS_DIR = if ($env:ANNOTATIONS_DIR) { $env:ANNOTATIONS_DIR } else { "data" }
$RESULTS_DIR = if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { "results" }
$CACHE_DIR = if ($env:CACHE_DIR) { $env:CACHE_DIR } else { "cache/embeddings" }

# Dataset configurations
$DATASETS = @{
    "music-avqa" = @{
        "annotations" = "$ANNOTATIONS_DIR/music-avqa/annotations.json"
        "data_root" = "$ANNOTATIONS_DIR/music-avqa"
    }
    "avqa" = @{
        "annotations" = "$ANNOTATIONS_DIR/avqa/annotations.csv"
        "data_root" = "$ANNOTATIONS_DIR/avqa"
    }
    "audiocaps" = @{
        "annotations" = "$ANNOTATIONS_DIR/audiocaps/annotations.json"
        "data_root" = "$ANNOTATIONS_DIR/audiocaps"
    }
}

# Interventions to test
$INTERVENTIONS = @("present", "masked", "swapped")

# Fusion types to test
$FUSION_TYPES = @("early", "late", "multimodal")

# Counters
$TOTAL_EXPERIMENTS = 0
$COMPLETED_EXPERIMENTS = 0
$FAILED_EXPERIMENTS = 0

Write-Host "========================================" -ForegroundColor Blue
Write-Host "Running All Baseline Experiments" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Calculate total number of experiments
foreach ($dataset in $DATASETS.Keys) {
    foreach ($intervention in $INTERVENTIONS) {
        foreach ($fusion in $FUSION_TYPES) {
            $TOTAL_EXPERIMENTS++
        }
    }
}

Write-Host "Total experiments to run: $TOTAL_EXPERIMENTS" -ForegroundColor Yellow
Write-Host ""

# Function to run a single experiment
function Run-Experiment {
    param(
        [string]$dataset,
        [string]$fusion,
        [string]$intervention,
        [string]$annotations_file,
        [string]$data_root
    )
    
    $current = $COMPLETED_EXPERIMENTS + 1
    Write-Host "[$current/$TOTAL_EXPERIMENTS] Running: " -NoNewline -ForegroundColor Blue
    Write-Host "$dataset " -NoNewline -ForegroundColor Green
    Write-Host "| $fusion " -NoNewline -ForegroundColor Green
    Write-Host "| $intervention" -ForegroundColor Green
    
    $args = @(
        "src/run_baselines.py",
        "--dataset", $dataset,
        "--fusion", $fusion,
        "--intervention", $intervention,
        "--annotations", $annotations_file,
        "--data-root", $data_root,
        "--results-dir", $RESULTS_DIR,
        "--cache-dir", $CACHE_DIR,
        "--swap-seed", "42"
    )
    
    try {
        $result = & python $args
        if ($LASTEXITCODE -eq 0) {
            $script:COMPLETED_EXPERIMENTS++
            Write-Host "✓ Completed" -ForegroundColor Green
            return $true
        } else {
            $script:FAILED_EXPERIMENTS++
            Write-Host "✗ Failed" -ForegroundColor Red
            return $false
        }
    } catch {
        $script:FAILED_EXPERIMENTS++
        Write-Host "✗ Failed: $_" -ForegroundColor Red
        return $false
    }
    Write-Host ""
}

# Run all experiments
foreach ($dataset in $DATASETS.Keys) {
    $annotations_file = $DATASETS[$dataset].annotations
    $data_root = $DATASETS[$dataset].data_root
    
    # Check if annotations file exists
    if (-not (Test-Path $annotations_file)) {
        Write-Host "Warning: Annotations file not found: $annotations_file" -ForegroundColor Yellow
        Write-Host "Skipping dataset: $dataset" -ForegroundColor Yellow
        Write-Host ""
        continue
    }
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
    Write-Host "Dataset: $dataset" -ForegroundColor Blue
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
    Write-Host ""
    
    foreach ($intervention in $INTERVENTIONS) {
        foreach ($fusion in $FUSION_TYPES) {
            Run-Experiment -dataset $dataset -fusion $fusion -intervention $intervention -annotations_file $annotations_file -data_root $data_root | Out-Null
        }
    }
    Write-Host ""
}

# Generate results table
Write-Host "========================================" -ForegroundColor Blue
Write-Host "Generating Results Table" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

$table_args = @(
    "src/evaluations/make_results_table.py",
    "--results-dir", $RESULTS_DIR,
    "--output-dir", "$RESULTS_DIR/tables",
    "--output-name", "results_table"
)

try {
    $result = & python $table_args
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Results table generated" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to generate results table" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Failed to generate results table: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Blue
Write-Host "Summary" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Blue
Write-Host "Completed: $COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS" -ForegroundColor Green
if ($FAILED_EXPERIMENTS -gt 0) {
    Write-Host "Failed: $FAILED_EXPERIMENTS" -ForegroundColor Red
}
Write-Host ""
Write-Host "Done. Results in $RESULTS_DIR" -ForegroundColor Green


