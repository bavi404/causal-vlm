# Helper script to run baseline evaluation

param(
    [Parameter(Mandatory=$false)]
    [string]$Dataset = "audiocaps",
    
    [Parameter(Mandatory=$false)]
    [string]$Fusion = "early",
    
    [Parameter(Mandatory=$false)]
    [string]$Intervention = "present",
    
    [Parameter(Mandatory=$false)]
    [string]$Annotations = "data/test_audiocaps/annotations.csv",
    
    [Parameter(Mandatory=$false)]
    [string]$DataRoot = "data/test_audiocaps",
    
    [Parameter(Mandatory=$false)]
    [string]$Device = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$ForceRecompute
)

Write-Host "Running Baseline Evaluation" -ForegroundColor Green
Write-Host "===========================" -ForegroundColor Green
Write-Host ""
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Fusion: $Fusion" -ForegroundColor Yellow
Write-Host "Intervention: $Intervention" -ForegroundColor Yellow
Write-Host "Annotations: $Annotations" -ForegroundColor Yellow
Write-Host "Data Root: $DataRoot" -ForegroundColor Yellow
Write-Host ""

# Build command
$cmd = "py -m src.run_baselines --dataset $Dataset --fusion $Fusion --intervention $Intervention --annotations $Annotations --data-root $DataRoot"

if ($Device) {
    $cmd += " --device $Device"
}

if ($ForceRecompute) {
    $cmd += " --force-recompute"
}

Write-Host "Command: $cmd" -ForegroundColor Cyan
Write-Host ""

# Run command
Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Evaluation complete! Check results in:" -ForegroundColor Green
    Write-Host "  results/$Dataset/${Fusion}_${Intervention}.csv" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Evaluation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
