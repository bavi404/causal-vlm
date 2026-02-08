"""Generate results table from all evaluation CSV files."""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import numpy as np


def find_result_csvs(results_dir: str = "results") -> List[Path]:
    """Find all CSV files in results directory, excluding tables subdirectory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    
    # Find CSV files but exclude tables subdirectory
    csv_files = [f for f in results_path.rglob("*.csv") 
                 if "tables" not in f.parts]
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    return csv_files


def parse_csv_path(csv_path: Path) -> Tuple[str, str, str]:
    """
    Parse dataset, fusion, and intervention from CSV file path.
    
    Expected format: results/{dataset}/{fusion}_{intervention}.csv
    
    Returns:
        Tuple of (dataset, fusion, intervention)
    """
    parts = csv_path.parts
    if len(parts) < 2:
        return None, None, None
    
    # Get dataset from parent directory
    dataset = parts[-2]
    
    # Get fusion and intervention from filename
    filename = csv_path.stem  # Remove .csv extension
    match = re.match(r'(.+?)_(.+)', filename)
    if match:
        fusion = match.group(1)
        intervention = match.group(2)
        return dataset, fusion, intervention
    
    return dataset, None, None


def load_metrics_from_csv(csv_path: Path) -> Dict[str, float]:
    """
    Load metrics from CSV file.
    
    Returns:
        Dictionary mapping metric names to values
        Format: {task_metric: value} or {metric: value}
    """
    metrics = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get('task', '')
                metric_name = row.get('metric', '')
                value_str = row.get('value', '')
                try:
                    value = float(value_str)
                    # Store as both task_metric and just metric for compatibility
                    if task:
                        full_name = f"{task}_{metric_name}"
                        metrics[full_name] = value
                    metrics[metric_name] = value
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
    
    return metrics


def collect_all_results(results_dir: str = "results") -> Dict:
    """
    Collect all results from CSV files.
    
    Returns:
        Nested dictionary: results[dataset][fusion][intervention][metric] = value
    """
    csv_files = find_result_csvs(results_dir)
    
    results = {}
    
    for csv_path in csv_files:
        dataset, fusion, intervention = parse_csv_path(csv_path)
        
        if not all([dataset, fusion, intervention]):
            print(f"Skipping {csv_path}: could not parse path")
            continue
        
        metrics = load_metrics_from_csv(csv_path)
        
        if dataset not in results:
            results[dataset] = {}
        if fusion not in results[dataset]:
            results[dataset][fusion] = {}
        if intervention not in results[dataset][fusion]:
            results[dataset][fusion][intervention] = {}
        
        results[dataset][fusion][intervention] = metrics
    
    return results


def select_primary_metric(metrics: Dict[str, float], metric_priority: List[str] = None) -> Optional[float]:
    """
    Select primary metric from available metrics.
    
    Priority order:
    1. retrieval_R@1 (for retrieval)
    2. qa_accuracy (for QA)
    3. retrieval_R@5
    4. qa_mean_similarity
    5. retrieval_mean_similarity
    """
    if metric_priority is None:
        metric_priority = [
            'retrieval_R@1',
            'qa_accuracy',
            'retrieval_R@5',
            'qa_mean_similarity',
            'retrieval_mean_similarity',
            'retrieval_R@10',
        ]
    
    for metric_name in metric_priority:
        if metric_name in metrics:
            return metrics[metric_name]
    
    # If no priority metric found, return first numeric value
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return value
    
    return None


def create_results_table(
    results: Dict,
    fusion_to_model: Optional[Dict[str, str]] = None,
    metric_priority: List[str] = None,
) -> pd.DataFrame:
    """
    Create results table from collected results.
    
    Args:
        results: Nested dictionary of results
        fusion_to_model: Mapping from fusion type to model name (e.g., {'early': 'Baseline'})
        metric_priority: Priority list of metrics to use
    
    Returns:
        DataFrame with formatted results table
    """
    if fusion_to_model is None:
        # Default mapping
        fusion_to_model = {
            'early': 'Baseline',
            'late': 'Proposed1',
            'multimodal': 'Proposed2',
            'transformer': 'Proposed2',  # Alias for transformer
        }
    
    # Collect all datasets and interventions
    all_datasets = sorted(set(results.keys()))
    all_interventions = set()
    all_fusions = set()
    
    for dataset_results in results.values():
        for fusion_results in dataset_results.values():
            all_fusions.update(fusion_results.keys())
            for intervention_results in fusion_results.values():
                all_interventions.update(intervention_results.keys())
    
    all_interventions = sorted(all_interventions)
    all_fusions = sorted(all_fusions)
    
    # Map fusion types to model names
    model_names = [fusion_to_model.get(f, f) for f in all_fusions]
    
    # Build table data
    table_rows = []
    row_labels = []
    
    # Add dataset rows
    for dataset in all_datasets:
        row_values = []
        
        for fusion in all_fusions:
            # Average across interventions for this dataset+fusion
            values = []
            for intervention in all_interventions:
                if (dataset in results and 
                    fusion in results[dataset] and 
                    intervention in results[dataset][fusion]):
                    metrics = results[dataset][fusion][intervention]
                    value = select_primary_metric(metrics, metric_priority)
                    if value is not None:
                        values.append(value)
            
            # Average for this fusion across interventions
            if values:
                avg_value = np.mean(values)
                row_values.append(avg_value)
            else:
                row_values.append(None)
        
        # Overall average across all fusions
        valid_values = [v for v in row_values if v is not None]
        if valid_values:
            row_values.append(np.mean(valid_values))
        else:
            row_values.append(None)
        
        table_rows.append(row_values)
        row_labels.append(dataset)
    
    # Add intervention rows
    for intervention in all_interventions:
        row_values = []
        
        for fusion in all_fusions:
            # Average across datasets for this fusion+intervention
            values = []
            for dataset in all_datasets:
                if (dataset in results and 
                    fusion in results[dataset] and 
                    intervention in results[dataset][fusion]):
                    metrics = results[dataset][fusion][intervention]
                    value = select_primary_metric(metrics, metric_priority)
                    if value is not None:
                        values.append(value)
            
            # Average for this fusion across datasets
            if values:
                avg_value = np.mean(values)
                row_values.append(avg_value)
            else:
                row_values.append(None)
        
        # Overall average across all fusions
        valid_values = [v for v in row_values if v is not None]
        if valid_values:
            row_values.append(np.mean(valid_values))
        else:
            row_values.append(None)
        
        table_rows.append(row_values)
        # Format intervention name nicely
        intervention_label = f"Audio {intervention.replace('_', ' ').title()}"
        row_labels.append(intervention_label)
    
    # Create DataFrame
    # Columns: each fusion type, then avg column
    columns = []
    for fusion in all_fusions:
        model_name = fusion_to_model.get(fusion, fusion)
        columns.append(model_name)
    columns.append('avg')
    
    # Create DataFrame
    df = pd.DataFrame(table_rows, columns=columns, index=row_labels)
    
    return df


def format_table_value(value: Optional[float], precision: int = 2) -> str:
    """Format table value for display."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def export_markdown_table(df: pd.DataFrame, output_path: Path):
    """Export table as Markdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write("| ")
        f.write(" | ".join([""] + list(df.columns)))
        f.write(" |\n")
        
        # Write separator
        f.write("| ")
        f.write(" | ".join(["---"] * (len(df.columns) + 1)))
        f.write(" |\n")
        
        # Write rows
        for idx, row in df.iterrows():
            f.write("| ")
            f.write(" | ".join([str(idx)] + [format_table_value(v) for v in row.values]))
            f.write(" |\n")
    
    print(f"Markdown table saved to: {output_path}")


def export_csv_table(df: pd.DataFrame, output_path: Path):
    """Export table as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path)
    print(f"CSV table saved to: {output_path}")


def export_png_table(df: pd.DataFrame, output_path: Path, figsize: Tuple[int, int] = (12, 8)):
    """Export table as PNG image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for idx, row in df.iterrows():
        row_data = [str(idx)] + [format_table_value(v) for v in row.values]
        table_data.append(row_data)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=[""] + list(df.columns),
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns) + 1):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(weight='bold', color='white', size=11)
        table[(0, i)].set_edgecolor('white')
        table[(0, i)].set_linewidth(1.5)
    
    # Identify dataset vs intervention rows
    num_datasets = 0
    for idx in df.index:
        if any(x in idx.lower() for x in ['present', 'masked', 'swapped']):
            break
        num_datasets += 1
    
    # Style dataset rows
    for i in range(1, num_datasets + 1):
        for j in range(len(df.columns) + 1):
            table[(i, j)].set_facecolor('#E8F5E9')
            table[(i, j)].set_edgecolor('white')
            table[(i, j)].set_linewidth(1)
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')
    
    # Style intervention rows
    for i in range(num_datasets + 1, len(table_data) + 1):
        for j in range(len(df.columns) + 1):
            table[(i, j)].set_facecolor('#FFF3E0')
            table[(i, j)].set_edgecolor('white')
            table[(i, j)].set_linewidth(1)
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')
    
    # Style avg column
    for i in range(len(table_data) + 1):
        table[(i, len(df.columns))].set_facecolor('#FFE0B2')
        table[(i, len(df.columns))].set_text_props(weight='bold')
    
    plt.title('Results Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print(f"PNG table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate results table from evaluation CSVs")
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing result CSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/tables',
        help='Directory to save output tables'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='results_table',
        help='Base name for output files (without extension)'
    )
    parser.add_argument(
        '--fusion-mapping',
        type=str,
        nargs='+',
        default=None,
        help='Mapping of fusion types to model names (e.g., early:Baseline late:Proposed1)'
    )
    parser.add_argument(
        '--metric-priority',
        type=str,
        nargs='+',
        default=None,
        help='Priority list of metrics to use (e.g., retrieval_R@1 qa_accuracy)'
    )
    
    args = parser.parse_args()
    
    # Parse fusion mapping
    fusion_to_model = None
    if args.fusion_mapping:
        fusion_to_model = {}
        for mapping in args.fusion_mapping:
            if ':' in mapping:
                fusion, model = mapping.split(':', 1)
                fusion_to_model[fusion] = model
    
    # Collect results
    print("Collecting results from CSV files...")
    results = collect_all_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found results for {len(results)} datasets")
    
    # Create table
    print("Creating results table...")
    df = create_results_table(
        results,
        fusion_to_model=fusion_to_model,
        metric_priority=args.metric_priority,
    )
    
    print("\nResults Table:")
    print(df.to_string())
    print()
    
    # Export in all formats
    output_dir = Path(args.output_dir)
    base_name = args.output_name
    
    export_markdown_table(df, output_dir / f"{base_name}.md")
    export_csv_table(df, output_dir / f"{base_name}.csv")
    export_png_table(df, output_dir / f"{base_name}.png")
    
    print("\nAll exports complete!")


if __name__ == '__main__':
    main()

