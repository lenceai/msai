#!/usr/bin/env python3
"""
Update the apply_post_training_pruning function in the notebook to fix NameError
"""

import json

notebook_path = '/home/lence/msai/UdaciSense/notebooks/02_compression.ipynb'

# Load the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# The correct function code
new_code = [
    "# Define a function to apply pruning and evaluate results\n",
    "def apply_post_training_pruning(config):\n",
    "    \"\"\"\n",
    "    Apply post-training pruning to a model with given pruning method and amount\n",
    "    \"\"\"\n",
    "    # Extract configuration parameters\n",
    "    pruning_method = config['pruning_method']\n",
    "    amount = config['amount']\n",
    "    device = config['device']\n",
    "\n",
    "    # Define unique experiment name given main parameters\n",
    "    experiment_name = f\"post_training/pruning/{pruning_method}_{amount}_{device}\"\n",
    "    experiment_name = experiment_name.replace('.', '-')\n",
    "    \n",
    "    # Create experiment subdirectories\n",
    "    os.makedirs(f\"../models/{experiment_name}\", exist_ok=True)\n",
    "    os.makedirs(f\"../results/{experiment_name}\", exist_ok=True)\n",
    "    \n",
    "    print(f\"Applying post-training pruning with method {pruning_method} and amount {amount:.2f}\")\n",
    "    \n",
    "    # Make a copy of the baseline model and move to specified device\n",
    "    orig_model = load_model(f\"../models/{baseline_model_name}/checkpoints/model.pth\").to(device)\n",
    "    orig_model.eval()\n",
    "    \n",
    "    # Apply post-training pruning\n",
    "    # TODO: IMPLEMENT THIS FUNCTION IN THE compression/ FOLDER \n",
    "    pruned_model = prune_model(orig_model, pruning_method, amount, config[\"modules_to_prune\"], config[\"custom_pruning_fn\"])\n",
    "    \n",
    "    # Save the pruned model\n",
    "    save_model(pruned_model, f\"../models/{experiment_name}/model.pth\")\n",
    "    \n",
    "    # Evaluate pruned model\n",
    "    metrics, confusion_matrix = evaluate_optimized_model(\n",
    "        pruned_model, \n",
    "        test_loader, \n",
    "        experiment_name,\n",
    "        class_names,\n",
    "        input_size,\n",
    "        device=device,\n",
    "    )\n",
    "    \n",
    "    # Compare with baseline model for performance differences\n",
    "    comparison_results = compare_optimized_model_to_baseline(\n",
    "        baseline_model,\n",
    "        pruned_model,\n",
    "        experiment_name,\n",
    "        test_loader,\n",
    "        class_names,\n",
    "        device=device,\n",
    "    )\n",
    "    \n",
    "    return pruned_model, comparison_results, experiment_name\n"
]

# Find and update the cell
updated = False
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def apply_post_training_pruning(config):' in source:
            cell['source'] = new_code
            updated = True
            print("✅ Updated apply_post_training_pruning cell")
            break

if updated:
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("✅ Notebook saved successfully")
else:
    print("❌ Could not find the cell to update")