import os
import pandas as pd
import json
import re
from datetime import datetime
import numpy as np

# List of sustainability-related keywords to filter content
SUSTAINABILITY_KEYWORDS = [
    'sustainability', 'sustainable', 'green', 'climate', 'carbon', 'emission', 'emissions',
    'renewable', 'clean energy', 'net zero', 'net-zero', 'environmental', 'csr', 
    'responsible', 'eco', 'pollution', 'recycle', 'recycling', 'biodiversity',
    'circular economy', 'energy efficiency', 'footprint', 'greenhouse', 'ghg',
    'impact', 'esg', 'social responsibility', 'natural resources', 'waste reduction',
    'conservation', 'solar', 'wind power', 'electric', 'low carbon', 'neutral', 'neutrality'
]

# Regular expression for detecting sustainability-related content
SUSTAINABILITY_PATTERN = re.compile(r'\b(' + '|'.join(SUSTAINABILITY_KEYWORDS) + r')\b', re.IGNORECASE)

def is_sustainability_related(text):
    """Check if text is related to sustainability using keywords."""
    if not isinstance(text, str):
        return False
    return bool(SUSTAINABILITY_PATTERN.search(text))

def parse_timestamp(timestamp_str):
    """Parse various timestamp formats into a standard format."""
    if not isinstance(timestamp_str, str):
        # If it's already a datetime object or NaT
        if isinstance(timestamp_str, pd.Timestamp) or pd.isna(timestamp_str):
            return timestamp_str
        # If it's a year as integer
        if isinstance(timestamp_str, (int, float)) and timestamp_str > 1900 and timestamp_str < 2100:
            try:
                return pd.Timestamp(int(timestamp_str), 1, 1)
            except:
                return pd.NaT
        return pd.NaT
    
    try:
        # Twitter timestamp format: "Mon Mar 04 10:22:27 +0000 2024"
        return pd.to_datetime(timestamp_str, errors='coerce')
    except:
        # Try common date formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y']:
            try:
                return pd.to_datetime(timestamp_str, format=fmt, errors='coerce')
            except:
                pass
    return pd.NaT

def create_engagement_metrics(row):
    """Create engagement metrics as a JSON string."""
    metrics = {}
    
    # Check if a url exists for the post
    if 'url' in row and pd.notna(row['url']):
        metrics['url'] = row['url']
    
    # Default engagement stats (can be enhanced if real metrics are available)
    if 'likes' in row and pd.notna(row['likes']):
        metrics['likes'] = row['likes']
    else:
        metrics['likes'] = 0
        
    if 'shares' in row and pd.notna(row['shares']):
        metrics['shares'] = row['shares']
    else:
        metrics['shares'] = 0
        
    if 'comments' in row and pd.notna(row['comments']):
        metrics['comments'] = row['comments']
    else:
        metrics['comments'] = 0
        
    return json.dumps(metrics)

def process_excel_file(file_path):
    """Process an Excel file and extract relevant data."""
    print(f"Processing {file_path}...")
    
    company_name = None
    claims_df = None
    company_info = {}
    
    try:
        # Read the Excel file
        xl = pd.ExcelFile(file_path)
        
        # Extract company name from filename if possible
        filename = os.path.basename(file_path)
        company_match = re.match(r'^(.*?)(?:_| merged| claims| -)', filename)
        if company_match:
            company_name = company_match.group(1).strip()
        
        # Process sheets
        if 'Company' in xl.sheet_names:
            company_df = pd.read_excel(file_path, sheet_name='Company')
            
            # Extract company name
            if 'Name' in company_df.columns and not company_name:
                company_name = company_df.loc[0, 'Name']
            elif company_df.shape[1] >= 2 and isinstance(company_df.columns[0], str):
                # For cases where company name is a column header
                if company_df.columns[0] == 'Name':
                    name_col = company_df.columns[1]
                    name_row = company_df[company_df[company_df.columns[0]] == 'Name']
                    if not name_row.empty:
                        company_name = name_row.iloc[0, 1]
            
            # Extract other company information for reference
            company_info = {'company': company_name}
            for field in ['Jurisdiction', 'Sector', 'Annual revenue', 'GHG Emissions']:
                if field in company_df.columns:
                    company_info[field.lower()] = company_df.loc[0, field]
                else:
                    # Check if it's in the first column with value in second column
                    field_row = company_df[company_df[company_df.columns[0]] == field]
                    if not field_row.empty:
                        company_info[field.lower()] = field_row.iloc[0, 1]
        
        # Process Claims sheet
        claims_sheet = None
        for sheet in xl.sheet_names:
            if sheet.startswith('Claim'):
                claims_sheet = sheet
                break
        
        if claims_sheet:
            claims_df = pd.read_excel(file_path, sheet_name=claims_sheet)
            
            # If company name not found yet, try to extract from claims
            if not company_name and 'Company' in claims_df.columns:
                company_name = claims_df['Company'].iloc[0]
            elif not company_name and 'Name' in claims_df.columns:
                company_name = claims_df['Name'].iloc[0]
            
            # Clean up and ensure company name is valid
            if not company_name or pd.isna(company_name):
                company_name = "Unknown Company"
        
        else:
            print(f"No Claims sheet found in {file_path}")
            return pd.DataFrame()  # Return empty DataFrame
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame
    
    if claims_df is None or claims_df.empty:
        print(f"No claims data found in {file_path}")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Map columns to dashboard format
    result_data = []
    
    # Identify column names for content, source, and timestamp
    content_col = None
    timestamp_col = None
    source_col = None
    
    # Check for content column
    for col in ['Description', 'full_text', 'text', 'description']:
        if col in claims_df.columns:
            content_col = col
            break
    
    # Check for timestamp column
    for col in ['created_at', 'Year', 'timestamp', 'date']:
        if col in claims_df.columns:
            timestamp_col = col
            break
    
    # Check for source column
    for col in ['Data source', 'Data Source', 'platform', 'source']:
        if col in claims_df.columns:
            source_col = col
            break
    
    if not content_col:
        print(f"Could not identify content column in {file_path}")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Process each row
    for idx, row in claims_df.iterrows():
        # Skip rows without content
        if content_col not in row or pd.isna(row[content_col]):
            continue

        content = str(row[content_col])
        
        # Filter for sustainability-related content
        if not is_sustainability_related(content):
            continue
            
        # Extract platform/source
        platform = "Unknown"
        if source_col and source_col in row and pd.notna(row[source_col]):
            platform = row[source_col]
        
        # Extract and standardize timestamp
        timestamp = None
        if timestamp_col and timestamp_col in row and pd.notna(row[timestamp_col]):
            timestamp = parse_timestamp(row[timestamp_col])
        
        if timestamp is None or pd.isna(timestamp):
            # Default to current date if no timestamp available
            timestamp = pd.Timestamp.now()
            
        # Create engagement metrics
        engagement_metrics = create_engagement_metrics(row)
        
        # Create comments list (empty for now as we don't have comments data)
        comments = json.dumps([])
        
        # Create data row
        result_data.append({
            'post_id': idx + 1,  # Generate sequential post ID
            'company': company_name,
            'platform': platform,
            'timestamp': timestamp,
            'content': content,
            'engagement_metrics': engagement_metrics,
            'comments': comments,
        })
    
    result_df = pd.DataFrame(result_data)
    
    # Add additional company info to the first row
    if not result_df.empty and company_info:
        for key, value in company_info.items():
            if key != 'company' and value is not None:  # Skip if already set or None
                result_df.at[0, f'company_{key}'] = value
    
    print(f"Extracted {len(result_df)} sustainability-related claims from {file_path}")
    return result_df

def main():
    """Main function to consolidate data from all Excel files."""
    print("Starting data consolidation...")
    
    # Path to the example data directory
    data_dir = 'example data'
    
    # Get all Excel files
    excel_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(f"Found {len(excel_files)} Excel files")
    
    # Process each file
    all_data = []
    for file_path in excel_files:
        df = process_excel_file(file_path)
        if not df.empty:
            all_data.append(df)
    
    # Combine all data into a single DataFrame
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data contains {len(combined_df)} rows")
        
        # Ensure timestamp is in the correct format (string)
        combined_df['timestamp'] = combined_df['timestamp'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') 
            if isinstance(x, pd.Timestamp) 
            else '2024-01-01 00:00:00'  # Default value for invalid timestamps
        )
        
        # Save to CSV
        output_file = 'consolidated_sustainability_data.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("No data was extracted from the Excel files")

if __name__ == "__main__":
    main() 