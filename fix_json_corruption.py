#!/usr/bin/env python3
"""
Fix the corrupted JSON in the notebook file
"""

file_path = '/home/lence/msai/UdaciSense/notebooks/02_compression.ipynb'

with open(file_path, 'r') as f:
    content = f.read()

# The corrupted section (raw code inserted into JSON list)
corrupted_part = """   "source": [
    "    # Define a function to apply pruning and evaluate results
    def apply_post_training_pruning(config):
        \"\"\"
        Apply post-training pruning to a model with given pruning method and amount
        \"\"\"
        # Extract configuration parameters
        pruning_method = config['pruning_method']
        amount = config['amount']
        device = config['device']

        # Define unique experiment name given main parameters
        experiment_name = f"post_training/pruning/{pruning_method}_{amount}_{device}"n","""

# The correct JSON structure
fixed_part = """   "source": [
    "# Define a function to apply pruning and evaluate results\\n",
    "def apply_post_training_pruning(config):\\n",
    "    \\"\\"\\"\\n",
    "    Apply post-training pruning to a model with given pruning method and amount\\n",
    "    \\"\\"\\"\\n",
    "    # Extract configuration parameters\\n",
    "    pruning_method = config['pruning_method']\\n",
    "    amount = config['amount']\\n",
    "    device = config['device']\\n",
    "\\n",
    "    # Define unique experiment name given main parameters\\n",
    "    experiment_name = f\\"post_training/pruning/{pruning_method}_{amount}_{device}\\"\\n","""

# Try to locate and replace
# We need to be careful with exact matching since whitespace might be tricky
# Let's find the start of the cell
start_marker = '"source": [\n    "    # Define a function'

if start_marker in content:
    # This looks like where the corruption starts
    # Find where the previous successful JSON structure ended
    pass

# Alternative: Rewrite the specific lines knowing the line numbers from the Read output
lines = content.splitlines()
# Based on read output:
# Line 483:    "source": [
# Line 484:     "    # Define a function to apply pruning and evaluate results
# ...
# Line 496:         experiment_name = f"post_training/pruning/{pruning_method}_{amount}_{device}"n",

# Let's verify the lines around 480-500
print(f"Checking lines 480-500:")
for i, line in enumerate(lines[480:500], 481):
    print(f"{i}: {line}")

# Reconstruct the valid JSON lines
# We'll replace the corrupted block with valid JSON strings
new_lines = []
skip = False
for i, line in enumerate(lines):
    if 'def apply_post_training_pruning(config):' in line and not line.strip().startswith('"'):
        # This is the start of the corruption
        new_lines.append('    "# Define a function to apply pruning and evaluate results\\n",')
        new_lines.append('    "def apply_post_training_pruning(config):\\n",')
        new_lines.append('    "    \\"\\"\\"\\n",')
        new_lines.append('    "    Apply post-training pruning to a model with given pruning method and amount\\n",')
        new_lines.append('    "    \\"\\"\\"\\n",')
        new_lines.append('    "    # Extract configuration parameters\\n",')
        new_lines.append('    "    pruning_method = config[\'pruning_method\']\\n",')
        new_lines.append('    "    amount = config[\'amount\']\\n",')
        new_lines.append('    "    device = config[\'device\']\\n",')
        new_lines.append('    "\\n",')
        new_lines.append('    "    # Define unique experiment name given main parameters\\n",')
        new_lines.append('    "    experiment_name = f\\"post_training/pruning/{pruning_method}_{amount}_{device}\\"\\n",')
        skip = True
    elif skip and 'experiment_name = experiment_name.replace' in line:
        skip = False
        new_lines.append(line)
    elif not skip:
        # Check for the line that starts the bad block (the comment line 484)
        if '    # Define a function to apply pruning and evaluate results' in line and not line.strip().endswith('\\n",'):
             # Skip this line, we added it above
             pass
        else:
            new_lines.append(line)

# Write back
with open(file_path, 'w') as f:
    f.write('\n'.join(new_lines))

print("✅ File fixed")