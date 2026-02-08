# PowerShell script to test framework with test.csv

Write-Host "Testing Framework with test.csv" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""

# Step 1: Convert CSV
Write-Host "Step 1: Converting test.csv to framework format..." -ForegroundColor Yellow
python convert_test_csv.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to convert CSV" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Running framework test (mock embeddings)..." -ForegroundColor Yellow
python test_framework.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Test failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=================================" -ForegroundColor Green
Write-Host "Test Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run full evaluation with actual model:" -ForegroundColor Cyan
Write-Host "python src/run_baselines.py --dataset audiocaps --fusion early --intervention present --annotations data/test_audiocaps/annotations.csv --data-root data/test_audiocaps" -ForegroundColor White
