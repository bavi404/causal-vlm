#!/bin/bash

# Bash script to test framework with test.csv

echo "Testing Framework with test.csv"
echo "================================="
echo ""

# Step 1: Convert CSV
echo "Step 1: Converting test.csv to framework format..."
python convert_test_csv.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert CSV"
    exit 1
fi

echo ""
echo "Step 2: Running framework test (mock embeddings)..."
python test_framework.py

if [ $? -ne 0 ]; then
    echo "Error: Test failed"
    exit 1
fi

echo ""
echo "================================="
echo "Test Complete!"
echo ""
echo "To run full evaluation with actual model:"
echo "python src/run_baselines.py --dataset audiocaps --fusion early --intervention present --annotations data/test_audiocaps/annotations.csv --data-root data/test_audiocaps"
