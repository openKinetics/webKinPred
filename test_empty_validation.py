#!/usr/bin/env python3
"""Test script to verify empty column validation works correctly."""

import pandas as pd
import sys
sys.path.append('/home/saleh/webKinPred')

from api.utils.validation_utils import validate_column_emptiness

# Test case 1: CSV with mostly empty substrates (like your example)
print("=== Test 1: CSV with mostly empty substrates ===")
data1 = {
    'Protein Sequence': [
        'MAKESTGFKPGSAKKGATLFKTRCQQCHTIEEGGPNKVGPNLHGIFGRHSGQVKGYSYTDANINKNVK',
        'MLWKRTCTRLIKPIAQPRGRLVRRSCYRYASTGTGSTDSSSQWLKYSVIASSATLFGYLFAKNLYSR',
        'ACDEFGHIKLMNPQRSTVWY'
    ],
    'Substrate': ['', '', '']  # All empty
}
df1 = pd.DataFrame(data1)
result1 = validate_column_emptiness(df1, 'Substrate')
print(f"Result: {result1}")

# Test case 2: CSV with some valid substrates
print("\n=== Test 2: CSV with some valid substrates ===")
data2 = {
    'Protein Sequence': [
        'MAKESTGFKPGSAKKGATLFKTRCQQCHTIEEGGPNKVGPNLHGIFGRHSGQVKGYSYTDANINKNVK',
        'MLWKRTCTRLIKPIAQPRGRLVRRSCYRYASTGTGSTDSSSQWLKYSVIASSATLFGYLFAKNLYSR',
        'ACDEFGHIKLMNPQRSTVWY'
    ],
    'Substrate': ['CC(C(=O)O)O', 'CC(C(=O)O)O', 'CC(C(=O)O)O']  # All valid
}
df2 = pd.DataFrame(data2)
result2 = validate_column_emptiness(df2, 'Substrate')
print(f"Result: {result2}")

# Test case 3: CSV with exactly 10% empty (should pass)
print("\n=== Test 3: CSV with exactly 10% empty (should pass) ===")
data3 = {
    'Protein Sequence': ['SEQ'] * 10,
    'Substrate': ['CC(C(=O)O)O'] * 9 + ['']  # 1 out of 10 empty = 10%
}
df3 = pd.DataFrame(data3)
result3 = validate_column_emptiness(df3, 'Substrate')
print(f"Result: {result3}")

# Test case 4: CSV with 11% empty (should fail)
print("\n=== Test 4: CSV with 11% empty (should fail) ===")
data4 = {
    'Protein Sequence': ['SEQ'] * 9,
    'Substrate': ['CC(C(=O)O)O'] * 8 + ['']  # 1 out of 9 empty = 11.1%
}
df4 = pd.DataFrame(data4)
result4 = validate_column_emptiness(df4, 'Substrate')
print(f"Result: {result4}")
