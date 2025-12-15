#!/usr/bin/env python3
"""
Fix the NameError in the Jupyter notebook by adding config parameter extraction
"""

import json

# Load the notebook
with open('/home/lence/msai/UdaciSense/notebooks/02_compression.ipynb', 'r') as f:
    notebook = json.load(f)

# Find and fix the apply_post_training_pruning function
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def apply_post_training_pruning(config):' in source:
            # Replace the problematic section
            old_lines = [
                "    \"\"\"\n",
                "    Apply post-training pruning to a model with given pruning method and amount\n",
                "    \"\"\"\n",
                "    # Define unique experiment name given main parameters\n"
            ]

            new_lines = [
                "    \"\"\"\n",
                "    Apply post-training pruning to a model with given pruning method and amount\n",
                "    \"\"\"\n",
                "    # Extract configuration parameters\n",
                "    pruning_method = config['pruning_method']\n",
                "    amount = config['amount']\n",
                "    device = config['device']\n",
                "\n",
                "    # Define unique experiment name given main parameters\n"
            ]

            # Replace in the source
            old_text = ''.join(old_lines)
            new_text = ''.join(new_lines)

            if old_text in source:
                source = source.replace(old_text, new_text)
                cell['source'] = source.split('\n')
                # Add back the newlines
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]

                print("✅ Fixed apply_post_training_pruning function")
                break

# Save the fixed notebook
with open('/home/lence/msai/UdaciSense/notebooks/02_compression.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Notebook fixed! The NameError should be resolved.")