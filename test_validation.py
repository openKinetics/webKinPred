#!/usr/bin/env python
"""Test the emptiness validation logic"""
import pandas as pd
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from api.utils.validation_utils import validate_column_emptiness

# Test case 1: CSV with empty substrates like your example
print("=" * 60)
print("Test 1: CSV with empty Substrate column (like your example)")
print("=" * 60)

data1 = {
    'Protein Sequence': [
        'MAKESTGFKPGSAKKGATLFKTRCQQCHTIEEGGPNKVGPNLHGIFGRHSGQVKGYSYTDANINKNVKWDEDSMSEYLTNPKKYIPGTKMAFAGLKKEKDRNDLITYMTKAAK',
        'MLWKRTCTRLIKPIAQPRGRLVRRSCYRYASTGTGSTDSSSQWLKYSVIASSATLFGYLFAKNLYSRETKEDLIEKLEMVKKIDPVNSTLKLSSLDSPDYLHDPVKIDKVVEDLKQVLGNKPENYSDAKSDLDAHSDTYFNTHHPSPEQRPRIILFPHTTEEVSKILKICHDNNMPVVPFSGGTSLEGHFLPTRIGDTITVDLSKFMNNVVKFDKLDLDITVQAGLPWEDLNDYLSDHGLMFGCDPGPGAQIGGCIANSCSGTNAYRYGTMKENIINMTIVLPDGTIVKTKKRPRKSSAGYNLNGLFVGSEGTLGIVTEATVKCHVKPKAETVAVVSFDTIKDAAACASNLTQSGIHLNAMELLDENMMKLINASESTDRCDWVEKPTMFFKIGGRSPNIVNALVDEVKAVAQLNHCNSFQFAKDDDEKLELWEARKVALWSVLDADKSKDKSAKIWTTDVAVPVSQFDKVIHETKKDMQASKLINAIVGHAGDGNFHAFIVYRTPEEHETCSQLVDRMVKRALNAEGTCTGEHGVGIGKREYLLEELGEAPVDLMRKIKLAIDPKRIMNPDKIFKTDPNEPANDYR'
    ],
    'Substrate': ['', '']  # Both empty
}
df1 = pd.DataFrame(data1)
error1 = validate_column_emptiness(df1, 'Substrate')
print(f"Result: {error1}")
print()

# Test case 2: CSV with some empty substrates (20%)
print("=" * 60)
print("Test 2: CSV with 20% empty substrates")
print("=" * 60)

data2 = {
    'Protein Sequence': ['SEQ1', 'SEQ2', 'SEQ3', 'SEQ4', 'SEQ5'],
    'Substrate': ['CC(C(=O)O)O', '', 'CCCC', 'CCCCC', 'CCCCCC']
}
df2 = pd.DataFrame(data2)
error2 = validate_column_emptiness(df2, 'Substrate')
print(f"Result: {error2}")
print()

# Test case 3: CSV with 5% empty substrates (should pass)
print("=" * 60)
print("Test 3: CSV with 5% empty substrates (should pass)")
print("=" * 60)

data3 = {
    'Protein Sequence': ['SEQ1', 'SEQ2', 'SEQ3', 'SEQ4', 'SEQ5', 'SEQ6', 'SEQ7', 'SEQ8', 'SEQ9', 'SEQ10',
                          'SEQ11', 'SEQ12', 'SEQ13', 'SEQ14', 'SEQ15', 'SEQ16', 'SEQ17', 'SEQ18', 'SEQ19', 'SEQ20'],
    'Substrate': ['SMILES'] * 19 + ['']  # Only 1 out of 20 empty = 5%
}
df3 = pd.DataFrame(data3)
error3 = validate_column_emptiness(df3, 'Substrate')
print(f"Result: {error3 if error3 else 'PASSED - No error'}")
print()

# Test case 4: Many empty rows (should list some)
print("=" * 60)
print("Test 4: CSV with 3 empty substrates out of 10")
print("=" * 60)

data4 = {
    'Protein Sequence': ['SEQ1', 'SEQ2', 'SEQ3', 'SEQ4', 'SEQ5', 'SEQ6', 'SEQ7', 'SEQ8', 'SEQ9', 'SEQ10'],
    'Substrate': ['SMILES', '', 'SMILES', '', 'SMILES', 'SMILES', '', 'SMILES', 'SMILES', 'SMILES']
}
df4 = pd.DataFrame(data4)
error4 = validate_column_emptiness(df4, 'Substrate')
print(f"Result: {error4}")
print()

# Test case 5: Empty Protein Sequence column
print("=" * 60)
print("Test 5: CSV with all empty Protein Sequences")
print("=" * 60)

data5 = {
    'Protein Sequence': ['', '', ''],
    'Substrate': ['SMILES', 'SMILES', 'SMILES']
}
df5 = pd.DataFrame(data5)
error5 = validate_column_emptiness(df5, 'Protein Sequence')
print(f"Result: {error5}")
print()
