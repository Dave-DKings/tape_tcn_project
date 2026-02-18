#!/usr/bin/env python3
"""
Quick test to check if there's a num_updates scope issue in the notebook cell
"""

import json

with open('phase_1_architecture_analysis.ipynb') as f:
    nb = json.load(f)

# Find the training cell
for cell in nb['cells']:
    if cell.get('id') == 'VSC-4435a9ad':
        source = ''.join(cell['source'])
        
        # Check indentation of key lines
        lines = source.split('\n')
        
        print("Checking stateful TCN training block structure...\n")
        
        in_stateful = False
        indent_level = 0
        
        for i, line in enumerate(lines):
            # Track when we enter stateful TCN block
            if 'elif architecture == \'stateful_tcn\':' in line:
                in_stateful = True
                indent_level = len(line) - len(line.lstrip())
                print(f"Line {i}: Stateful TCN block starts (indent={indent_level})")
                
            elif in_stateful:
                current_indent = len(line) - len(line.lstrip())
                
                # Check if we've exited the block
                if line.strip() and not line.strip().startswith('#') and current_indent <= indent_level:
                    print(f"Line {i}: Stateful TCN block ends (found: {line.strip()[:50]})")
                    in_stateful = False
                
                # Check for num_updates definition and usage
                elif 'num_updates' in line and not line.strip().startswith('#'):
                    print(f"Line {i}: num_updates referenced (indent={current_indent})")
                    print(f"         {line.rstrip()}")
        
        print("\nDone!")
        break
