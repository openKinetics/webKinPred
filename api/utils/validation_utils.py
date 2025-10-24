"""
Validation utility functions for input processing.
These functions handle specific validation tasks following single responsibility principle.
"""
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from api.utils.convert_to_mol import convert_to_mol

try:
    from webKinPred.config_docker import MODEL_LIMITS, SERVER_LIMIT
except ImportError:
    from webKinPred.config_local import MODEL_LIMITS, SERVER_LIMIT


def safe_convert_to_mol(value: Any) -> Optional[Any]:
    """
    Safely convert a value to molecule format with robust NaN/empty string handling.
    
    Args:
        value: Input value to convert
        
    Returns:
        Converted molecule object or None if invalid/empty
    """
    # Robust guard for NaN / non-str / empty strings
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    
    if not isinstance(value, str):
        return None
    
    sanitized_value = value.strip()
    if not sanitized_value or sanitized_value in ["None", "NaN", "nan"]:
        return None
        
    return convert_to_mol(sanitized_value)


def validate_csv_structure(dataframe: pd.DataFrame) -> Optional[str]:
    """
    Validate that CSV has required columns for substrate/protein validation.
    
    Args:
        dataframe: Pandas DataFrame to validate
        
    Returns:
        Error message if validation fails, None if valid
    """
    has_substrate_col = "Substrate" in dataframe.columns
    has_multi_substrate_cols = "Substrates" in dataframe.columns and "Products" in dataframe.columns
    has_protein_col = "Protein Sequence" in dataframe.columns
    
    if not (has_substrate_col or has_multi_substrate_cols):
        return 'CSV must contain "Substrate" column OR "Substrates" and "Products" columns'
    
    if not has_protein_col:
        return 'CSV must contain a "Protein Sequence" column'
    
    return None


def validate_single_substrate_schema(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate substrates using single-substrate schema (Substrate column).
    
    Args:
        dataframe: DataFrame containing Substrate column
        
    Returns:
        List of validation errors
    """
    invalid_substrates = []
    val_to_output = {}
    
    for i, val in enumerate(dataframe["Substrate"]):
        row_num = i + 1
        
        if val in val_to_output:
            if val_to_output[val] == "OK":
                continue
            temp = val_to_output[val].copy()
            temp["row"] = row_num
            invalid_substrates.append(temp)
        elif safe_convert_to_mol(val) is None:
            error_entry = {
                "row": row_num,
                "value": val,
                "reason": "Invalid SMILES/InChI",
            }
            val_to_output[val] = error_entry
            invalid_substrates.append(error_entry)
        else:
            val_to_output[val] = "OK"
    
    return invalid_substrates


def validate_multi_substrate_schema(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate substrates using multi-substrate schema (Substrates and Products columns).
    
    Args:
        dataframe: DataFrame containing Substrates and Products columns
        
    Returns:
        List of validation errors
    """
    invalid_substrates = []
    
    for i, (subs, prods) in enumerate(zip(dataframe["Substrates"], dataframe["Products"])):
        row_num = i + 1
        invalid_tokens = []
        
        # Validate each substrate token
        for substrate in str(subs).split(";"):
            token = substrate.strip()
            if token and safe_convert_to_mol(token) is None:
                invalid_tokens.append(token)
        
        # Validate each product token
        for product in str(prods).split(";"):
            token = product.strip()
            if token and safe_convert_to_mol(token) is None:
                invalid_tokens.append(token)
        
        if invalid_tokens:
            invalid_substrates.append({
                "row": row_num,
                "value": invalid_tokens,
                "reason": "Invalid substrate/product SMILES/InChI",
            })
    
    return invalid_substrates


def validate_substrates(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate substrate data based on available columns in the dataframe.
    
    Args:
        dataframe: DataFrame to validate substrates from
        
    Returns:
        List of substrate validation errors
    """
    if "Substrate" in dataframe.columns:
        return validate_single_substrate_schema(dataframe)
    elif "Substrates" in dataframe.columns and "Products" in dataframe.columns:
        return validate_multi_substrate_schema(dataframe)
    else:
        return []


def validate_protein_sequence_characters(sequence: str) -> List[str]:
    """
    Validate that protein sequence contains only valid amino acid characters.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        List of invalid characters found in sequence
    """
    valid_alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    return sorted({char for char in sequence if char not in valid_alphabet})


def calculate_sequence_length_violations(sequence_length: int) -> Dict[str, int]:
    """
    Calculate which models would reject a sequence based on length limits.
    
    Args:
        sequence_length: Length of the protein sequence
        
    Returns:
        Dictionary mapping model names to violation count (0 or 1)
    """
    violations = {}
    
    if sequence_length > SERVER_LIMIT:
        violations["Server"] = 1
    else:
        violations["Server"] = 0
    
    for model_name, limit in MODEL_LIMITS.items():
        if sequence_length > limit:
            violations[model_name] = 1
        else:
            violations[model_name] = 0
    
    return violations


def validate_protein_sequences(dataframe: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Validate protein sequences for character validity and length constraints.
    
    Args:
        dataframe: DataFrame containing Protein Sequence column
        
    Returns:
        Tuple of (invalid_proteins_list, aggregated_length_violations)
    """
    invalid_proteins = []
    total_length_violations = {model: 0 for model in MODEL_LIMITS}
    total_length_violations["Server"] = 0
    
    for i, sequence in enumerate(dataframe["Protein Sequence"]):
        row_num = i + 1
        
        # Check for empty sequences
        if not isinstance(sequence, str) or len(sequence.strip()) == 0:
            invalid_proteins.append({
                "row": row_num,
                "value": sequence or "",
                "reason": "Empty sequence"
            })
            continue
        
        cleaned_sequence = sequence.strip()
        sequence_length = len(cleaned_sequence)
        
        # Aggregate length violations
        length_violations = calculate_sequence_length_violations(sequence_length)
        for model, violation_count in length_violations.items():
            total_length_violations[model] += violation_count
        
        # Check for invalid characters
        invalid_chars = validate_protein_sequence_characters(cleaned_sequence)
        if invalid_chars:
            invalid_proteins.append({
                "row": row_num,
                "value": cleaned_sequence,
                "invalid_chars": invalid_chars,
                "reason": "Invalid characters in sequence",
            })
    
    return invalid_proteins, total_length_violations


def clean_data_for_json(data: Any) -> Any:
    """
    Recursively clean data to make it JSON-serializable.
    Converts pandas NaN values to string "NaN".
    
    Args:
        data: Data to clean (can be nested lists/dicts)
        
    Returns:
        JSON-serializable version of the data
    """
    if isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif pd.isna(data):
        return "NaN"
    else:
        return data


def parse_csv_file(file) -> pd.DataFrame:
    """
    Parse and clean a CSV file from request.
    
    Args:
        file: File object from Django request
        
    Returns:
        Cleaned pandas DataFrame
        
    Raises:
        Exception: If CSV parsing fails
    """
    dataframe = pd.read_csv(file)
    return dataframe.dropna(how="all")  # Remove empty rows


def validate_file_format(file, allowed_extensions: List[str] = None) -> Optional[str]:
    """
    Validate file format based on allowed extensions.
    
    Args:
        file: File object from request
        allowed_extensions: List of allowed extensions (defaults to ['.csv'])
        
    Returns:
        Error message if invalid, None if valid
    """
    if allowed_extensions is None:
        allowed_extensions = ['.csv']
    
    if not any(file.name.lower().endswith(ext) for ext in allowed_extensions):
        ext_str = ", ".join(allowed_extensions)
        return f"File format not supported. Please upload a file with extension: {ext_str}"
    
    return None


def validate_required_columns(dataframe: pd.DataFrame, required_columns: List[str]) -> Optional[str]:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        dataframe: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Error message if validation fails, None if valid
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        return f'Missing required columns: {", ".join(missing_columns)}'
    
    return None


def validate_column_emptiness(dataframe: pd.DataFrame, column_name: str, max_empty_percent: float = 0.1) -> Optional[str]:
    """
    Validate that a column doesn't have too many empty values.
    
    Args:
        dataframe: DataFrame to validate
        column_name: Name of the column to check
        max_empty_percent: Maximum allowed percentage of empty rows (default 0.1 = 10%)
        
    Returns:
        Error message if validation fails, None if valid
    """
    if column_name not in dataframe.columns:
        return None  # Column doesn't exist, will be caught by other validation
    
    total_rows = len(dataframe)
    if total_rows == 0:
        return None
    
    # Check for empty values (NaN, None, empty strings, whitespace-only strings)
    empty_mask = dataframe[column_name].isna() | (dataframe[column_name].astype(str).str.strip() == '')
    empty_rows = dataframe[empty_mask].index + 2  # +2 because: +1 for header, +1 for 0-indexed to 1-indexed
    num_empty = len(empty_rows)
    empty_percent = num_empty / total_rows
    
    # If all or most rows are empty (>90%)
    if empty_percent > 0.9:
        return f'{column_name} column is empty'
    
    # If more than max_empty_percent are empty
    if empty_percent > max_empty_percent:
        if num_empty <= 10:
            row_list = ', '.join(map(str, empty_rows))
            return f'Rows {row_list} have no {column_name.lower()} value'
        else:
            return f'{num_empty} rows have no {column_name.lower()} value (>{max_empty_percent*100:.0f}% of data)'
    
    return None


def validate_column_emptiness(dataframe: pd.DataFrame, column_name: str, max_empty_percent: float = 0.1) -> Optional[str]:
    """
    Validate that a column doesn't have too many empty values.
    
    Args:
        dataframe: DataFrame to validate
        column_name: Name of the column to check
        max_empty_percent: Maximum allowed percentage of empty rows (default 0.1 = 10%)
        
    Returns:
        Error message if validation fails, None if valid
    """
    if column_name not in dataframe.columns:
        return None  # Column doesn't exist, will be caught by other validation
    
    total_rows = len(dataframe)
    if total_rows == 0:
        return None
    
    # Check for empty values (NaN, None, empty strings, whitespace-only strings)
    empty_mask = dataframe[column_name].isna() | (dataframe[column_name].astype(str).str.strip() == '')
    empty_rows = dataframe[empty_mask].index + 2  # +2 because: +1 for header, +1 for 0-indexed to 1-indexed
    num_empty = len(empty_rows)
    empty_percent = num_empty / total_rows
    
    # If all or most rows are empty (>90%)
    if empty_percent > 0.9:
        return f'{column_name} column is empty'
    
    # If more than max_empty_percent are empty
    if empty_percent > max_empty_percent:
        if num_empty <= 10:
            # List specific rows if not too many
            row_list = ', '.join(map(str, empty_rows.tolist()))
            return f'Rows {row_list} have no {column_name} value'
        else:
            # Too many to list individually
            return f'{num_empty} rows ({empty_percent*100:.1f}%) have no {column_name} value'
    
    return None
