
import json
import re
import os

def convert_md_to_ipynb(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Extract image definitions
    # Regex for [imageX]: <data:image/png;base64,...>
    # Note: The file uses < and > around the data URI
    img_def_pattern = re.compile(r'^\[(image\d+)\]:\s*<(data:image/[a-zA-Z]+;base64,[^>]+)>\s*$', re.MULTILINE)
    
    img_map = {}
    for match in img_def_pattern.finditer(content):
        img_id = match.group(1)
        data_uri = match.group(2)
        img_map[img_id] = data_uri

    # 2. Remove image definitions from content
    content_no_defs = img_def_pattern.sub('', content)

    # 3. Inline images
    # Pattern: ![][imageX]
    # We replace it with ![imageX](data_uri)
    def replace_img_ref(match):
        img_id = match.group(1)
        if img_id in img_map:
            return f"![{img_id}]({img_map[img_id]})"
        return match.group(0)

    # The file uses ![][image1]
    img_ref_pattern = re.compile(r'!\[\]\[(image\d+)\]')
    content_inlined = img_ref_pattern.sub(replace_img_ref, content_no_defs)

    # 4. Split into cells by headers
    # We want to split whenever we see a line starting with #, but keep the # line in the following cell.
    # We can just split by lines and accumulate.
    
    cells = []
    current_source = []

    lines = content_inlined.split('\n')
    
    for line in lines:
        # Check if line is a header
        if re.match(r'^#+\s', line):
            # If we have accumulated current source, push it as a cell
            if current_source:
                # Strip trailing empty lines
                while current_source and current_source[-1].strip() == "":
                    current_source.pop()
                if current_source:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": "\n".join(current_source)
                    })
            current_source = [line]
        else:
            current_source.append(line)
            
    # Push the last cell
    if current_source:
         while current_source and current_source[-1].strip() == "":
                    current_source.pop()
         if current_source:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": "\n".join(current_source)
            })

    # 5. Construct Notebook JSON
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Successfully converted {input_path} to {output_path}")

if __name__ == "__main__":
    input_file = r"c:\Users\Owner\new_project\adaptive_portfolio_rl\RL Portfolio Optimization Feature Engineering.md"
    output_file = r"c:\Users\Owner\new_project\adaptive_portfolio_rl\RL_Portfolio_Optimization_Feature_Engineering.ipynb"
    convert_md_to_ipynb(input_file, output_file)
