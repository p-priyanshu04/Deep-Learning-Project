import json
import sys

def extract_notebook(path, out_path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, cell in enumerate(nb['cells']):
            f.write(f"# --- CELL {i} ({cell['cell_type']}) ---\n")
            if cell['cell_type'] == 'markdown':
                for line in cell.get('source', []):
                    f.write(f"# {line}")
                f.write("\n\n")
            elif cell['cell_type'] == 'code':
                for line in cell.get('source', []):
                    f.write(line)
                f.write("\n\n")

if __name__ == '__main__':
    extract_notebook(sys.argv[1], sys.argv[2])
