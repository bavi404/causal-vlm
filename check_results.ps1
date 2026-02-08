# Script to check for results and show their location

Write-Host "Checking for Results..." -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host ""

$resultsDir = "results"
if (Test-Path $resultsDir) {
    Write-Host "Results directory found: $resultsDir" -ForegroundColor Green
    Write-Host ""
    
    # Find all CSV files
    $csvFiles = Get-ChildItem -Path $resultsDir -Filter "*.csv" -Recurse
    if ($csvFiles.Count -gt 0) {
        Write-Host "Found $($csvFiles.Count) result file(s):" -ForegroundColor Yellow
        Write-Host ""
        foreach ($file in $csvFiles) {
            $relativePath = $file.FullName.Replace((Get-Location).Path + "\", "")
            Write-Host "  - $relativePath" -ForegroundColor Cyan
        }
        
        # Check for tables
        $tablesDir = Join-Path $resultsDir "tables"
        if (Test-Path $tablesDir) {
            Write-Host ""
            Write-Host "Aggregated tables:" -ForegroundColor Yellow
            Get-ChildItem -Path $tablesDir | Where-Object { $_.Extension -match '\.(csv|md|png)$' } | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor Cyan
            }
        }
    } else {
        Write-Host "No CSV result files found in $resultsDir" -ForegroundColor Yellow
    }
} else {
    Write-Host "Results directory not found: $resultsDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Results will be saved here when you run evaluations:" -ForegroundColor Cyan
    Write-Host "  results/{dataset}/{fusion}_{intervention}.csv" -ForegroundColor White
    Write-Host ""
    Write-Host "To generate results, run:" -ForegroundColor Yellow
    $cmd = "py src/run_baselines.py --dataset audiocaps --fusion early --intervention present --annotations data/test_audiocaps/annotations.csv --data-root data/test_audiocaps"
    Write-Host "  $cmd" -ForegroundColor White
}

Write-Host ""
