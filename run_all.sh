#!/bin/bash

# Script to run all baseline experiments across datasets, interventions, and fusion types

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths (can be overridden with environment variables)
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CACHE_DIR="${CACHE_DIR:-cache/embeddings}"

# Dataset configurations
# Format: dataset_name:annotations_file:data_root
declare -A DATASETS=(
    ["music-avqa"]="${ANNOTATIONS_DIR}/music-avqa/annotations.json:${ANNOTATIONS_DIR}/music-avqa"
    ["avqa"]="${ANNOTATIONS_DIR}/avqa/annotations.csv:${ANNOTATIONS_DIR}/avqa"
    ["audiocaps"]="${ANNOTATIONS_DIR}/audiocaps/annotations.json:${ANNOTATIONS_DIR}/audiocaps"
)

# Interventions to test
INTERVENTIONS=("present" "masked" "swapped")

# Fusion types to test
FUSION_TYPES=("early" "late" "multimodal")

# Counters
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running All Baseline Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Calculate total number of experiments
for dataset in "${!DATASETS[@]}"; do
    for intervention in "${INTERVENTIONS[@]}"; do
        for fusion in "${FUSION_TYPES[@]}"; do
            ((TOTAL_EXPERIMENTS++))
        done
    done
done

echo -e "${YELLOW}Total experiments to run: ${TOTAL_EXPERIMENTS}${NC}"
echo ""

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local fusion=$2
    local intervention=$3
    local annotations_file=$4
    local data_root=$5
    
    echo -e "${BLUE}[$((COMPLETED_EXPERIMENTS + 1))/${TOTAL_EXPERIMENTS}]${NC} Running: ${GREEN}${dataset}${NC} | ${GREEN}${fusion}${NC} | ${GREEN}${intervention}${NC}"
    
    if python src/run_baselines.py \
        --dataset "${dataset}" \
        --fusion "${fusion}" \
        --intervention "${intervention}" \
        --annotations "${annotations_file}" \
        --data-root "${data_root}" \
        --results-dir "${RESULTS_DIR}" \
        --cache-dir "${CACHE_DIR}" \
        --swap-seed 42; then
        ((COMPLETED_EXPERIMENTS++))
        echo -e "${GREEN}✓ Completed${NC}"
    else
        ((FAILED_EXPERIMENTS++))
        echo -e "${RED}✗ Failed${NC}"
        return 1
    fi
    echo ""
}

# Run all experiments
for dataset in "${!DATASETS[@]}"; do
    # Parse dataset configuration
    IFS=':' read -r annotations_file data_root <<< "${DATASETS[$dataset]}"
    
    # Check if annotations file exists
    if [ ! -f "${annotations_file}" ]; then
        echo -e "${YELLOW}Warning: Annotations file not found: ${annotations_file}${NC}"
        echo -e "${YELLOW}Skipping dataset: ${dataset}${NC}"
        echo ""
        continue
    fi
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Dataset: ${dataset}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    for intervention in "${INTERVENTIONS[@]}"; do
        for fusion in "${FUSION_TYPES[@]}"; do
            run_experiment "${dataset}" "${fusion}" "${intervention}" "${annotations_file}" "${data_root}" || true
        done
    done
    echo ""
done

# Generate results table
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Results Table${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if python src/evaluations/make_results_table.py \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${RESULTS_DIR}/tables" \
    --output-name "results_table"; then
    echo -e "${GREEN}✓ Results table generated${NC}"
else
    echo -e "${RED}✗ Failed to generate results table${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Completed: ${GREEN}${COMPLETED_EXPERIMENTS}${NC}/${TOTAL_EXPERIMENTS}"
if [ ${FAILED_EXPERIMENTS} -gt 0 ]; then
    echo -e "Failed: ${RED}${FAILED_EXPERIMENTS}${NC}"
fi
echo ""
echo -e "${GREEN}Done. Results in ${RESULTS_DIR}${NC}"


