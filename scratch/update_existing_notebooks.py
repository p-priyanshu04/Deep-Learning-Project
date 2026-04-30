import nbformat as nbf
import os

def insert_code(nb_path, import_str, code_str, title):
    nb = nbf.read(nb_path, as_version=4)
    new_cells = []
    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and import_str in cell.source:
            new_cells.append(nbf.v4.new_markdown_cell(f"### {title}"))
            new_cells.append(nbf.v4.new_code_cell(code_str))
            cell.source = cell.source.replace(import_str, f"# {import_str} (Replaced with inline code)")
            new_cells.append(cell)
            modified = True
        else:
            new_cells.append(cell)
    
    if modified:
        nb.cells = new_cells
        nbf.write(nb, nb_path)
        print(f"Updated {nb_path}")
    else:
        print(f"Import string '{import_str}' not found in {nb_path}")

with open('../utils/data_utils.py', 'r', encoding='utf-8') as f:
    data_utils_code = f.read()
insert_code('../notebooks/01_Data_Preprocessing.ipynb', 'from utils.data_utils import *', data_utils_code, 'Data Utils Functions')

with open('../utils/model_utils.py', 'r', encoding='utf-8') as f:
    model_utils_code = f.read()
insert_code('../notebooks/02_Model_Implementation.ipynb', 'from utils.model_utils import RNNAVG, RNNATT, RNNPOA', model_utils_code, 'Model Utils Functions')

with open('../utils/train_utils.py', 'r', encoding='utf-8') as f:
    train_utils_code = f.read()
insert_code('../notebooks/03_Training_and_Experiments.ipynb', 'from utils.train_utils import train_model, evaluate, compute_metrics', train_utils_code, 'Train Utils Functions')
