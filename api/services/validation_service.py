"""
Validation service that orchestrates input validation workflows.
"""
from typing import Dict, Any
from api.utils.validation_utils import (
    parse_csv_file,
    validate_csv_structure,
    validate_substrates,
    validate_protein_sequences,
    clean_data_for_json,
)

def validate_input_file(file) -> Dict[str, Any]:
    """
    Validate an uploaded CSV file for substrate and protein sequence data.
    
    Args:
        file: Uploaded file object from Django request
        
    Returns:
        Dictionary containing validation results or error information
    """
    # Parse CSV file
    try:
        dataframe = parse_csv_file(file)
    except Exception as e:
        return {
            "error": f"Invalid CSV format: {str(e)}",
            "status_code": 400
        }
    
    # Validate CSV structure
    structure_error = validate_csv_structure(dataframe)
    if structure_error:
        return {
            "error": structure_error,
            "status_code": 400
        }
    
    # Validate substrates
    invalid_substrates = validate_substrates(dataframe)
    
    # Validate protein sequences
    invalid_proteins, length_violations = validate_protein_sequences(dataframe)
    
    # Clean data for JSON serialization
    return {
        "invalid_substrates": clean_data_for_json(invalid_substrates),
        "invalid_proteins": clean_data_for_json(invalid_proteins),
        "length_violations": clean_data_for_json(length_violations),
        "status_code": 200
    }
