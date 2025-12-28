import pandas as pd
from rdkit import Chem
import argparse
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import warnings

from rdkit.Chem import rdmolfiles
import numpy as np
from utils import mol2atoms, mol2coords, write_lmdb, parse_nmr, check_nmr_text

from base_logger import Logger
logger = Logger(log_name="filter_nmr_data").get_logger()


def parallel_apply(data, func, *args, **kwargs):
    with Pool() as pool:
        func_with_args = partial(func, *args, **kwargs)
        result = pool.map(func_with_args, tqdm(data))
    return result


def count_chiral_centers(mol):
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

def smi2mol(smi, add_hs=True):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mol = Chem.MolFromSmiles(smi)
            if add_hs and mol is not None:
                mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
            if len(w) > 0:
                return None
            return mol
    except Exception as e:
        return None

def has_radical(mol):
    return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())

def has_isotope(mol):
    return any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())

def is_valid_smiles(smi):
    return '*' not in smi and '.' not in smi

def is_valid_molecule(mol):
    return not (has_radical(mol) or has_isotope(mol))

def create_atoms_target_mask(atoms, atoms_equi_class, nmr_type):
    np.random.seed(42)
    if nmr_type == 'H':
        return np.array([1 if atom == nmr_type else 0 for atom in atoms], dtype=np.int32)
    elif nmr_type == 'C':
        mask = np.zeros(len(atoms), dtype=np.int32)
        unique_classes = set(atoms_equi_class[i] for i, atom in enumerate(atoms) if atom == 'C')
        for equi_class in unique_classes:
            indices = [i for i, cls in enumerate(atoms_equi_class) if cls == equi_class and atoms[i] == 'C']
            if indices:
                chosen_index = np.random.choice(indices)
                mask[chosen_index] = 1
        return mask

def fill_atoms_target(mask, extracted_nmr):
    target = np.zeros_like(mask, dtype=np.float32)
    nmr_values = np.array(extracted_nmr, dtype=np.float32)
    target[mask == 1] = nmr_values[:np.sum(mask)]
    return target

def check_element(mol, allowed_atoms=None):
    if allowed_atoms is None:
        return True
    return all(atom.GetSymbol() in allowed_atoms for atom in mol.GetAtoms())

def get_equi_class(mol):
    equi_class = rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)
    return np.array(equi_class).astype(np.int16)

def filter_data(input_path, nmr_type, max_chiral_centers, max_atoms, allowed_atoms, filter_warning, h_gap):
    
    df = pd.read_parquet(input_path)
    logger.info(f'-- num before filter: {len(df)}')

    # Filter by NMR type
    nmr_mapping = {
        "H": "1H NMR",
        "C": "13C NMR"
    }
    df = df[df['nmr_type'] == nmr_mapping[nmr_type]]
    # top 10000
    # df = df.head(10000)
    logger.info(f'-- num after nmr type filter: {len(df)}')

    # Filter by smiles
    df['validity'] = parallel_apply(df['smiles'], is_valid_smiles)
    df = df[df['validity'] == True]
    logger.info(f'-- num after smiles filter: {len(df)}')
    
    # Convert smiles to mol
    df['mol'] = parallel_apply(df['smiles'], smi2mol)
    df = df[pd.notnull(df['mol'])]
    logger.info(f'-- num after smiles to mol filter: {len(df)}')
    
    # Filter by radical and isotope
    df['validity'] = parallel_apply(df['mol'], is_valid_molecule)
    df = df[df['validity'] == True]
    logger.info(f'-- num after radical and isotope filter: {len(df)}')
    
    # Filter by chiral centers
    df['n_chiral_centers'] = parallel_apply(df['mol'], count_chiral_centers)
    df = df[df['n_chiral_centers'] <= max_chiral_centers]
    logger.info(f'-- num after chiral centers filter: {len(df)}')
    
    # check NMR
    df['nmr_check'] = parallel_apply(df['nmr_shift'], check_nmr_text, nmr_type=nmr_type)
    df = df[df['nmr_check'] == True]
    logger.info(f'-- num after check nmr filter: {len(df)}')
    
    # parser NMR
    import ast
    df["nmr_processed"] = parallel_apply(df["nmr_processed"], ast.literal_eval)
    df["extracted_shifts"] = parallel_apply(df["nmr_processed"], parse_nmr, nmr_type=nmr_type, h_gap=h_gap)
    del df["nmr_processed"], df['nmr_shift'], df['nmr_note'], df['location_in_page_smiles'], df['location_in_page_para']
    # print(df.head(3))
    # stop
    df = df[pd.notnull(df["extracted_shifts"])]
    logger.info(f'-- num after parse nmr filter: {len(df)}')
    
    
    ### From Here: only necessary for forward model
    # Filter by atom count
    df['atoms'] = parallel_apply(df['mol'], mol2atoms)
    df = df[pd.notnull(df['atoms'])]
    df = df[df['atoms'].apply(lambda x: len(x) <= max_atoms)]
    logger.info(f'-- num after atom count filter: {len(df)}')
    
    # Filter by element
    df['check_element'] = parallel_apply(df['mol'], check_element, allowed_atoms=allowed_atoms)
    df = df[df['check_element'] == True]
    logger.info(f'-- num after check element filter: {len(df)}')
    
    # Filter by shifts count
    df['atoms_equi_class'] = parallel_apply(df['mol'], get_equi_class)
    df['atoms_target_mask'] = df.swifter.apply(
        lambda row: create_atoms_target_mask(row['atoms'], row['atoms_equi_class'], nmr_type),
        axis=1
    )
    df['count_mask'] = df['atoms_target_mask'].apply(lambda x: np.sum(x))
    df['count_shifts'] = df["extracted_shifts"].apply(lambda x: len(x))
    df = df[df['count_mask'] == df['count_shifts']]
    df['atoms_target'] = df.swifter.apply(
        lambda row: fill_atoms_target(row['atoms_target_mask'], row["extracted_shifts"]),
        axis=1
    )
    logger.info(f'-- num after shifts count filter: {len(df)}')
    
    # Get coordinates
    df['coordinates'] = parallel_apply(df['mol'], mol2coords, filter_warning=filter_warning)
    df = df[pd.notnull(df['coordinates'])]
    logger.info(f'-- num after coordinates filter: {len(df)}')
    
    # Save filtered data
    final_data = []
    for i in range(len(df)):
        final_data.append({
            "smiles": df.iloc[i]['smiles'],
            "atoms": df.iloc[i]['atoms'],
            "coordinates": df.iloc[i]['coordinates'],
            "atoms_target_mask": df.iloc[i]['atoms_target_mask'],
            "atoms_target": df.iloc[i]['atoms_target'],
            "nmr_frequency": df.iloc[i]['nmr_frequency'],
            "nmr_solvent": df.iloc[i]['nmr_solvent'],
        })
        
    logger.info(f'-- num final: {len(df)}')
    
    output_path = input_path.replace(".parquet", f"_{nmr_type}_max_chiral_{max_chiral_centers}_max_atoms_{max_atoms}_ele_{'_'.join(allowed_atoms) if allowed_atoms else 'All'}_filtered.lmdb")
    write_lmdb(final_data, output_path)
    logger.info(f"Filtered data saved to {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter NMR data from a .parquet file.")
    parser.add_argument("--input", '-i', default="./data/raw_data/NMRexp_10to24_1_0811.parquet", help="Path to the input .parquet file.")
    parser.add_argument("--nmr_type", '-t', default='H', choices=["H", "C"], help="NMR type to filter (1H NMR or 13C NMR).")
    parser.add_argument("--max_chiral_centers", '-c', type=int, default=1, help="Maximum number of chiral centers allowed.")
    parser.add_argument("--max_atoms", '-m', type=int, default=70, help="Maximum number of atoms allowed.")
    parser.add_argument("--allowed_atoms", '-a', default="C,H,O,N,S,P,F,Cl", help="Allowed atoms for filtering.")
    parser.add_argument("--filter_warning", '-f', default=True, help="Whether to filter out rdkit warnings.")
    parser.add_argument("--h_gap", type=float, default=0.2, help="Maximum allowed gap between chemical shifts for 1H NMR.")
    args = parser.parse_args()

    allowed_atoms = args.allowed_atoms
    if allowed_atoms is not None:
        allowed_atoms = [atom.strip() for atom in allowed_atoms.split(",")]
    
    logger.info(f'-- input: {args.input}')
    logger.info(f'-- nmr type: {args.nmr_type}')
    logger.info(f'-- max chiral centers: {args.max_chiral_centers}')
    logger.info(f'-- max atoms: {args.max_atoms}')
    logger.info(f'-- allowed atoms: {allowed_atoms}')
    logger.info(f'-- filter warning: {args.filter_warning}')
    logger.info(f'-- h gap: {args.h_gap}')
    
    filter_data(args.input, args.nmr_type, args.max_chiral_centers, args.max_atoms, allowed_atoms, args.filter_warning, args.h_gap)

# python filter_nmr_data.py -t H && python filter_nmr_data.py -t C