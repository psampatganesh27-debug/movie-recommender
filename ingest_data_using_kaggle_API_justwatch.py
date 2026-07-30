import pandas as pd
import os

print("--- Starting Database Filtering Pipeline ---")

# 1. Load the massive master dataset (adjust filename if yours varies slightly)
csv_path = "movies_master.csv"

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path} in your project directory.")
    exit()

print("Reading master file (this may take a moment due to file size)...")
df = pd.read_csv(csv_path, low_memory=False)

# 2. Drop records completely missing crucial text or IDs
df = df.dropna(subset=['id', 'title', 'overview'])

# 3. Clean and filter by relevance to ensure premium recommendations
# We filter out movies with 0 votes or completely non-existent popularity scores
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)

# 4. Extract Indian Cinema (Hindi, Telugu, Tamil, Malayalam, Kannada, Bengali)
indian_languages = ['hi', 'te', 'ta', 'ml', 'kn', 'bn']
indian_movies = df[df['original_language'].isin(indian_languages)]

# 5. Extract Global Cinema (Hollywood / International) with at least 5 user reviews
global_movies = df[(~df['original_language'].isin(indian_languages)) & (df['vote_count'] >= 5)]

# 6. Merge them together to form your high-quality global catalog
filtered_df = pd.concat([indian_movies, global_movies]).drop_duplicates(subset=['id'])

# 7. Cap the total database to the top ~60,000 most popular movies for speed efficiency
filtered_df = filtered_df.sort_values(by='popularity', ascending=False).head(60000)

# Save the final optimized dataset to your folder
output_path = "movies_cleaned.csv"
# Select ONLY the 4 critical columns needed for our recommendation and streaming platform engine
# In this dataset, 'id' is already the correct live TMDB identification number!
final_columns = ['id', 'title', 'overview', 'vote_average']
optimized_df = filtered_df[final_columns]

# Overwrite the clean file with ONLY our core data matrix
output_path = "movies_cleaned.csv"
optimized_df.to_csv(output_path, index=False)

print("\n===============================================")
print(f"SUCCESS: Filtered dataset saved as '{output_path}'")
print(f"Total Movies Extracted: {len(optimized_df):,}")
print(f"Kept Essential Columns: {', '.join(final_columns)}")
print("===============================================")

