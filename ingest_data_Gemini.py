import pandas as pd
import os

print("--- Starting Clean Core Data Pipeline ---")

master_path = "movies_master.csv"

if not os.path.exists(master_path):
    print(f"Error: '{master_path}' missing from directory.")
    exit()

print("Loading raw master database...")
df_master = pd.read_csv(master_path, low_memory=False)

# Clean master dataset columns
df_master = df_master.dropna(subset=['title', 'overview'])
df_master['popularity'] = pd.to_numeric(df_master['popularity'], errors='coerce').fillna(0)
df_master['vote_average'] = pd.to_numeric(df_master['vote_average'], errors='coerce').fillna(0)

print("Sorting purely by global popularity and slicing top 60,000 records...")
final_catalog = df_master.sort_values(by='popularity', ascending=False).head(60000)

# Export the clean catalog without the broken streaming column
output_path = "movies_cleaned.csv"
final_catalog[['title', 'overview', 'vote_average']].to_csv(output_path, index=False)

print(f"\nSUCCESS: Clean catalog saved to '{output_path}' with {len(final_catalog):,} rows!")