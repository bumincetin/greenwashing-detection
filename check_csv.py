import pandas as pd

# Read the CSV file
df = pd.read_csv('consolidated_sustainability_data.csv')

# Print basic information
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Print the first 2 rows
print("\nFirst 2 rows:")
print(df.iloc[:2].to_string())

# Check platform values
print("\nPlatform values distribution:")
platform_counts = df['platform'].value_counts()
for platform, count in platform_counts.items():
    print(f"  {platform}: {count} ({count/len(df)*100:.1f}%)")

# Check company distribution
print("\nCompany distribution (top 5):")
company_counts = df['company'].value_counts().head(5)
for company, count in company_counts.items():
    print(f"  {company}: {count} ({count/len(df)*100:.1f}%)")

# Sample of content and engagement metrics
print("\nSample content (first row):")
print(df.iloc[0]['content'][:200] + "..." if len(df.iloc[0]['content']) > 200 else df.iloc[0]['content'])

print("\nSample engagement metrics (first row):")
print(df.iloc[0]['engagement_metrics'])

print("\nSample comments (first row):")
print(df.iloc[0]['comments']) 