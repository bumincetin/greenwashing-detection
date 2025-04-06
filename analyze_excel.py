import os
import pandas as pd
import json

# List all Excel files in the directory
excel_files = [f for f in os.listdir('example data') if f.endswith('.xlsx')]
print(f'Found {len(excel_files)} Excel files in the directory.')

# Function to examine an Excel file
def examine_excel(file_path):
    print(f'\n\n===== EXAMINING: {file_path} =====')
    # Read Excel file and get sheet names
    try:
        xl = pd.ExcelFile(file_path)
        sheets = xl.sheet_names
        print(f'Sheets found: {sheets}')
        
        # Examine each sheet
        for sheet in sheets:
            print(f'\n--- Sheet: {sheet} ---')
            try:
                # Read a few rows to understand structure
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                
                # Display columns and data types
                print(f'Columns ({len(df.columns)}): {list(df.columns)}')
                print(f'Data types: \n{df.dtypes}')
                
                # Show first row as sample
                if not df.empty:
                    print('\nSample row:')
                    first_row = df.iloc[0].to_dict()
                    for key, value in first_row.items():
                        # Truncate long strings for display
                        if isinstance(value, str) and len(value) > 100:
                            print(f'{key}: {value[:100]}...')
                        else:
                            print(f'{key}: {value}')
                            
                # Check row count
                full_df = pd.read_excel(file_path, sheet_name=sheet)
                print(f'\nTotal rows: {len(full_df)}')
                
            except Exception as e:
                print(f'Error examining sheet {sheet}: {str(e)}')
    
    except Exception as e:
        print(f'Error examining file: {str(e)}')

# Files to examine (including different data sources)
files_to_examine = [
    'AIB Group_Example.xlsx',  # Already examined
    'BBC claims - merged.xlsx',  # Potentially different format
    'HSBC_report.xlsx',  # Potentially a report format
    'SAP_insigai.xlsx',  # Potentially from a different source
    'Carnival Corp Merged.xlsx'  # Another format to check
]

for file in files_to_examine:
    file_path = os.path.join('example data', file)
    examine_excel(file_path)

print('\n\nAnalysis complete.') 