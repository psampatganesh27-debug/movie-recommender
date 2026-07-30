import pandas as pd
import os

print("--- Starting Unified Global Popularity Pipeline ---")

master_path = "movies_master.csv"
stream_path = "streaming_data.csv"

if not os.path.exists(master_path) or not os.path.exists(stream_path):
    print("Error: Required files missing from project directory.")
    exit()

print("Loading raw master database...")
df_master = pd.read_csv(master_path, low_memory=False)

print("Loading offline streaming platform table...")
df_stream = pd.read_csv(stream_path, low_memory=False)

# 1. Clean the master dataset
df_master = df_master.dropna(subset=['title', 'overview'])
df_master['popularity'] = pd.to_numeric(df_master['popularity'], errors='coerce').fillna(0)
df_master['vote_average'] = pd.to_numeric(df_master['vote_average'], errors='coerce').fillna(0)

# 2. Normalize titles to lowercase to guarantee exact text matching links
df_master['title_clean'] = df_master['title'].astype(str).str.lower().str.strip()
df_stream['title_clean'] = df_stream['Title'].astype(str).str.lower().str.strip()

print("Processing streaming availability flags...")
def build_platform_string(row):
    platforms = []
    if row.get('Netflix') == 1: platforms.append('Netflix')
    if row.get('Prime Video') == 1: platforms.append('Amazon Prime Video')
    if row.get('Disney+') == 1: platforms.append('Disney+')
    if row.get('Hulu') == 1: platforms.append('Hulu')
    return ", ".join(platforms) if platforms else "Available to Rent/Buy or In Theaters"

df_stream['streaming_info'] = df_stream.apply(build_platform_string, axis=1)
df_stream_sliced = df_stream[['title_clean', 'streaming_info']].drop_duplicates(subset=['title_clean'])

print("Merging tables purely by title matching...")
# Merge data frames cleanly without any language-based splitting
merged_df = pd.merge(df_master, df_stream_sliced, on='title_clean', how='left')
merged_df['streaming_info'] = merged_df['streaming_info'].fillna("Available to Rent/Buy / Check Local Streaming")

print("Sorting purely by global popularity and slicing top 60,000 records...")
# This sorts the entire combined dataset globally, with no language favoritism
final_catalog = merged_df.sort_values(by='popularity', ascending=False).head(60000)

# Export the clean unified catalog
output_path = "movies_cleaned.csv"
final_catalog[['title', 'overview', 'vote_average', 'streaming_info']].to_csv(output_path, index=False)

print(f"\nSUCCESS: Unified local catalog saved to '{output_path}'")
print(f"Total Database Rows: {len(final_catalog):,}")
print("=======================================================")