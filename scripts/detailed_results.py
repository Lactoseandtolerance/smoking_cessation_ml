"""Extract more detailed results from notebook"""
import nbformat

# Load the notebook
with open('/Users/professornirvar/data mining/smoking_cessation_ml/notebooks/04_modeling.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print("=== EXTRACTING DETAILED RESULTS FROM NOTEBOOK ===\n")

# Find key outputs by execution count
key_cells = {
    20: "Train/Val/Test Split",
    21: "Logistic Regression Training",
    22: "Random Forest Training",
    23: "XGBoost Training",
    24: "Evaluation Summary",
    25: "Model Comparison Table",
    28: "Feature Importance"
}

for exec_count, description in key_cells.items():
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.get('execution_count') == exec_count:
            print(f"\n{'='*70}")
            print(f"{description} (Execution #{exec_count})")
            print('='*70)
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        print(output['text'].strip())
                    elif output.get('output_type') == 'stream' and 'text' in output:
                        print(output['text'].strip())
            break

print("\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)
