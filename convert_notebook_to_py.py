import json
import re
import os

def convert_notebook_to_py(notebook_path, output_path):
    """
    Convert a Jupyter notebook to a Python script while preserving content and structure.
    
    Args:
        notebook_path: Path to the Jupyter notebook (.ipynb file)
        output_path: Path where the Python script will be saved
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Create the Python script content
    python_content = []
    
    # Add header
    python_content.append('#!/usr/bin/env python')
    python_content.append('# -*- coding: utf-8 -*-')
    python_content.append('"""')
    python_content.append('Electricity Price Forecasting with LSTM')
    python_content.append('"""')
    python_content.append('')
    
    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # Convert markdown to Python comments
            content = cell['source']
            if isinstance(content, list):
                content = ''.join(content)
            
            # Split into lines and add comment markers
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    python_content.append(f'# {line}')
                else:
                    python_content.append('#')
            python_content.append('')
            
        elif cell['cell_type'] == 'code':
            # Add code as is
            content = cell['source']
            if isinstance(content, list):
                content = ''.join(content)
            
            # Add code with proper indentation
            lines = content.split('\n')
            for line in lines:
                python_content.append(line)
            python_content.append('')
    
    # Write the Python script
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(python_content))
    
    print(f"Notebook converted to Python script: {output_path}")

if __name__ == "__main__":
    # Define paths
    notebook_path = "electricity_price_forecast_lstm_pipeline.ipynb"
    output_path = "electricity_price_forecast_lstm_pipeline.py"
    
    # Convert notebook to Python script
    convert_notebook_to_py(notebook_path, output_path) 