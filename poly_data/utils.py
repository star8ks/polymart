import json
from poly_utils.google_utils import get_spreadsheet
import pandas as pd 
import os
from logan import Logan

def get_sheet_df(read_only=None):
    """
    Get sheet data with optional read-only mode
    
    Args:
        read_only (bool): If None, auto-detects based on credentials availability
    """
    all = 'All Markets'

    # Auto-detect read-only mode if not specified
    if read_only is None:
        creds_file = 'credentials.json' if os.path.exists('credentials.json') else '../credentials.json'
        read_only = not os.path.exists(creds_file)
        if read_only:
            Logan.info(
                "No credentials found, using read-only mode",
                namespace="poly_data.utils"
            )

    try:
        spreadsheet = get_spreadsheet(read_only=read_only)
    except FileNotFoundError:
        Logan.info(
            "No credentials found, falling back to read-only mode",
            namespace="poly_data.utils"
        )
        spreadsheet = get_spreadsheet(read_only=True)

    # Read directly from All Markets - no more filtering at read time
    wk = spreadsheet.worksheet(all)
    result = pd.DataFrame(wk.get_all_records())
    result = result[result['question'] != ""].reset_index(drop=True)

    wk_p = spreadsheet.worksheet('Hyperparameters')
    records = wk_p.get_all_records()
    hyperparams, current_type = {}, None

    for r in records:
        # Update current_type only when we have a non-empty type value
        # Handle both string and NaN values from pandas
        type_value = r['type']
        if type_value and str(type_value).strip() and str(type_value) != 'nan':
            current_type = str(type_value).strip()
        
        # Skip rows where we don't have a current_type set
        if current_type:
            # Convert numeric values to appropriate types
            value = r['value']
            try:
                # Try to convert to float if it's numeric
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    value = float(value)
                elif isinstance(value, (int, float)):
                    value = float(value)
            except (ValueError, TypeError):
                pass  # Keep as string if conversion fails
            
            hyperparams.setdefault(current_type, {})[r['param']] = value

    return result, hyperparams
