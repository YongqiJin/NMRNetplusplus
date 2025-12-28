import io
import sys
import contextlib
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import lmdb
import pickle
import os
from typing import Dict, List, Union
from tqdm import tqdm


def parse_c_text(text):
    pattern = r"\d+\.\d+-\d+\.\d+"
    
    pattern = r"(-?\d+(?:\.\d+)?)(?:\s*\(([^)]*)\))?\s*(?=,|$)"
    matches = re.finditer(pattern, text)

    remaining_text = re.sub(pattern + r"(,)?", "", text).strip()
    
    extracted = []
    for match in matches:
        num_str = match.group(1)
        content = match.group(2)
        if '.' not in num_str:
            return None, text
    
        extracted.append((float(num_str), content))

    remaining_text = remaining_text.replace(",", "").strip()

    return extracted, remaining_text

# extracted, remaining_text = parse_c_text("169.6, 146.1-145.9 (m), 145.6, 143.7-143.3 (m), 142.5-142.1 (m), 140.0-139.6 (m), 138.8-138.5 (m), 136.3-136.0 (m), 135.1, 130.0, 127.9, 57.5, 40.0, 22.6, 21.5")
# print(extracted)
# print(remaining_text)

def parse_h_text(text):
    pattern = r"(-?\d+(?:\.\d+)?)(?:\s*-\s*(-?\d+(?:\.\d+)?))?(?:\s*\(([^)]*)\))\s*(?=,|$)"
    # pattern = r"(-?\d+(?:[.,]\d+)?)(?:\s*-\s*(-?\d+(?:[.,]\d+)?))?(?:\s*\(([^)]*)\))\s*(?=,|$)"
    matches = re.finditer(pattern, text)

    # (num1, num2, content)
    extracted = []
    for match in matches:
        num1_str = match.group(1)
        num2_str = match.group(2) if match.group(2) else None
        if '.' not in num1_str:
            return None, text
        if num2_str and '.' not in num2_str:
            return None, text
        num1 = float(num1_str)
        num2 = float(num2_str) if num2_str else num1
        content = match.group(3)
        
        extracted.append((num1, num2, content))

    remaining_text = re.sub(pattern + r"(,)?", "", text).strip()
    remaining_text = remaining_text.replace(",", "").strip()

    return extracted, remaining_text

# extracted, remaining_text = parse_h_text("7.02 (d, J = 8.8 Hz, 2H), 6.82 (d, J = 8.8 Hz, 2H), 4.49 (dd, J = 14.8, 1.2 Hz, 1H), 4.08 (dd, J = 14.8 Hz, 1H), 3.89 (dd, J = 8.4, 6.0 Hz, 1H), 3.79 (s, 3H), 3.69 (dd, J = 8.4, 2.0 Hz, 1H), 3.36,-3.42 (m, 1H), 1.28 (s, 12H), 1.24 (d, J = 7.2 Hz, 3H)")
# print(extracted)
# print(remaining_text)


def parse_h_shifts(text):
    mode = text.split(',')[0]
    
    nH_match = re.search(r"\b(\d+)\s*H\b", text)
    nH = int(nH_match.group(1)) if nH_match else None
    text = re.sub(r"\b\d+\s*H\b", "", text).strip()
    
    j_matches = re.findall(r"=\s*([\d.\s,Hz]+)", text)
    j_values = []
    for match in j_matches:
        values = [float(re.sub(r"\s*Hz", "", v.strip())) for v in match.split(',') if re.match(r"^\d+(\.\d+)?\s*(Hz)?$", v.strip())] # 没处理 1. 2Hz bug
        j_values.extend(values)
    
    return mode, j_values, nH

def check_nmr_text(text, nmr_type):
    if nmr_type == "H":
        extract, remaining_text = parse_h_text(text)
        if remaining_text:
            return False
        if not is_monotonic([item[0] + item[1] for item in extract]):
            return False
        if any(item[0] + item[1] < 0 for item in extract) and is_monotonic([abs(item[0] + item[1]) for item in extract]):
            return False
    elif nmr_type == "C":
        extract, remaining_text = parse_c_text(text)
        if remaining_text:
            return False
        if not is_monotonic([item[0] for item in extract]):
            return False
        if any(item[0] < 0 for item in extract) and is_monotonic([abs(item[0]) for item in extract]):
            return None
    else:
        raise ValueError(f"Unsupported NMR type: {nmr_type}")
    return True

def parse_nmr(extract, nmr_type, h_min=-1, h_max=15, h_gap=0.5, c_min=-50, c_max=250):
    if nmr_type == "H":
        h_nmr = []
        for item in extract:
            nH, shift1, shift2 = item[2], item[3], item[4]
            try:
                assert nH[-1] == 'H'
                nH = int(nH[:-1])
            except:
                return None
            # large gap between shifts
            if abs(shift1 - shift2) > h_gap:
                return None
            # shift out of range
            if not (h_min < shift1 < h_max) or not (h_min < shift2 < h_max):
                return None
            h_nmr += [(shift1 + shift2) / 2] * nH
        return h_nmr
    elif nmr_type == "C":
        c_nmr = []
        for item in extract:
            shift = item[0]
            if not (c_min < shift < c_max):
                return None
            c_nmr.append(shift)
        return c_nmr
    else:
        raise ValueError(f"Unsupported NMR type: {nmr_type}")
    
def is_monotonic(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)) or \
           all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))

def mol2atoms(mol):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(atoms) == 0:
        return None
    return atoms


def mol2coords(mol, seed=42, mode='fast', filter_warning=True):
    '''
    Convert a molecule to 3D coordinates, optionally filtering molecules with RDKit warnings (e.g., UFFTYPER issues).

    :param mol: RDKit molecule object.
    :param seed: Random seed for conformer generation.
    :param mode: 'fast' or 'heavy' mode for embedding.
    :param filter_warning: If True, filter out molecules with UFFTYPER or similar warnings.
    :return: Coordinates (np.ndarray) or None if failed or filtered.
    '''
    stderr_buffer = io.StringIO()

    # Redirect stderr to catch RDKit warnings
    with contextlib.redirect_stderr(stderr_buffer):
        try:
            # will random generate conformer with seed equal to -1. else fixed random seed.
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            
            if filter_warning:
                stderr_output = stderr_buffer.getvalue()
            if 'UFFTYPER' in stderr_output:
                return None
            
            if res == 0:
                try:
                    # some conformer can not use MMFF optimize
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
                except:
                    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            ## for fast test... ignore this ###
            elif res == -1 and mode == 'heavy':
                AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
                try:
                    # some conformer can not use MMFF optimize
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
                except:
                    AllChem.Compute2DCoords(mol)
                    coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                    coordinates = coordinates_2d
            else:
                return None
        except:
            return None

    return coordinates


def load_lmdb(path):
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    txn_read = env.begin(write=False)

    data = {}
    with txn_read.cursor() as cursor:
        for key, value in cursor:
            data[key] = pickle.loads(value)

    env.close()
    return data


def write_lmdb(data: Union[Dict, List], path: str):
    """
    Writes data to an LMDB database.
    """
    try:
        os.remove(path)
        print("Remove existing lmdb: {}".format(os.path.abspath(path)))
    except:
        pass
    env_new = lmdb.open(
        path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        # max_readers=1,
        map_size=int(1e12),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    if isinstance(data, list):
        num = len(data)
        bit = len(str(num))  # Ensure enough digits to represent all keys
        for index in tqdm(range(len(data))):
            inner_output = pickle.dumps(data[index], protocol=-1)
            txn_write.put(str(i).zfill(bit).encode(), inner_output)
            i += 1
            if i % 100 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
        txn_write.commit()
        env_new.close()
    elif isinstance(data, dict):
        for key in tqdm(data.keys()):
            inner_output = pickle.dumps(data[key], protocol=-1)
            txn_write.put(key, inner_output)
            i += 1
            if i % 100 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
        txn_write.commit()
        env_new.close()
    else:
        raise ValueError("Data type not supported: {}".format(type(data)))
    
    print("Write to lmdb: {}".format(os.path.abspath(path)))