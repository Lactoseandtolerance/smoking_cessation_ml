"""Extract results from the executed notebook"""
import pandas as pd
import json

# The notebook has these variables available in memory
# Let's read the notebook output to extract results

import nbformat

# Load the notebook
with open('/Users/professornirvar/data mining/smoking_cessation_ml/notebooks/04_modeling.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Find the comparison table output (cell 22, execution count 25)
print("=== SEARCHING FOR MODEL COMPARISON RESULTS ===\n")

for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and cell.get('execution_count') == 25:
        print(f"Found cell {i+1} (execution count 25):")
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    print(output['text'])
                elif output.get('output_type') == 'stream' and 'text' in output:
                    print(output['text'])
        print("\n" + "="*60 + "\n")

# Also check for other important outputs
print("\n=== CHECKING DATA LOADING CELL ===")
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and cell.get('execution_count') == 17:
        print(f"Found data loading cell (execution count 17):")
        if 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    print(output['text'])
                elif output.get('output_type') == 'stream' and 'text' in output:
                    print(output['text'])
        break
