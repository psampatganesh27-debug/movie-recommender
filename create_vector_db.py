
#ONLINE VERSION

'''
import os
import time
import pandas as pd
import chromadb
from dotenv import load_dotenv
from google import genai

print("--- Initializing Rate-Resistant Vector Database Builder ---")

# 1. Load your API key securely
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()

# 2. Initialize the client
ai_client = genai.Client()

# 3. Load your clean 60k dataset
csv_path = "movies_cleaned.csv"
if not os.path.exists(csv_path):
    print(f"Error: Run ingest_data.py first to create {csv_path}")
    exit()

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} movies cleanly from CSV.")

# 4. Connect to your local ChromaDB instance
chroma_client = chromadb.PersistentClient(path="chroma_storage")

collection = chroma_client.get_or_create_collection(
    name="global_movies_recommender",
    metadata={"hnsw:space": "cosine"}
)

print("\nStarting batch vectorization with automatic rate-limit backing...")

batch_size = 50
total_movies = len(df)

# We use time.sleep between batches to keep the free tier calm
cooldown_seconds = 4

for i in range(0, total_movies, batch_size):
    batch_df = df.iloc[i : i + batch_size]
    
    ids = batch_df['id'].astype(str).tolist()
    documents = batch_df['overview'].tolist()
    metadatas = batch_df[['title', 'vote_average']].to_dict(orient='records')
    
    # We set up an internal retry loop to handle 429 quota exhaustion gracefully
    retries = 5
    wait_time = 60  # Initial seconds to wait if throttled
    
    while retries > 0:
        try:
            # Generate the text embeddings
            response = ai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=documents
            )
            
            embeddings = [e.values for e in response.embeddings]
            
            # Save right into your ChromaDB instance
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Successfully indexed rows {i} to {min(i + batch_size, total_movies)} / {total_movies}")
            
            # Small forced break between standard iterations to stay under the RPM limit
            time.sleep(cooldown_seconds)
            break  # Success! Break the retry loop and move to the next batch
            
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"\n Rate limit hit at index {i}. Backing off for {wait_time} seconds...")
                time.sleep(wait_time)
                retries -= 1
                wait_time *= 2  # Exponentially increase the pause time if it happens repeatedly
            else:
                print(f"Skipped batch starting at index {i} due to critical error: {e}")
                break  # Don't retry for syntax or structural errors

print("\n=======================================================")
print("SUCCESS: Local Vector Database built and saved to 'chroma_storage/'!")
print("=======================================================")

'''

#OFFLINE VERSION

import os
import time
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("--- Initializing Lightning-Fast LOCAL Vector Database Builder ---")

# 1. Load your clean 60k dataset
csv_path = "movies_cleaned.csv"
if not os.path.exists(csv_path):
    print(f"Error: Run ingest_data.py first to create {csv_path}")
    exit()

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} movies cleanly from CSV.")

# 2. Load the local embedding model (It downloads once, then runs 100% offline)
print("Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Connect to your local ChromaDB instance
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_or_create_collection(
    name="global_movies_recommender",
    metadata={"hnsw:space": "cosine"}
)

print("\nStarting high-speed offline vectorization...")
start_time = time.time()

# Since we are running locally, we can jack the batch size up to 500 for maximum speed!
batch_size = 500
total_movies = len(df)

for i in range(0, total_movies, batch_size):
    batch_df = df.iloc[i : i + batch_size]
    
    # FIX: Generate an absolute unique synthetic ID using the title combined with the row index number
    ids = [f"{row['title']}_{idx}" for idx, row in batch_df.iterrows()]
    
    documents = batch_df['overview'].tolist()
    #metadatas = batch_df[['title', 'vote_average', 'streaming_info']].to_dict(orient='records')
    # Remove 'streaming_info' from this line since now we will use gemini to get this info
    metadatas = batch_df[['title', 'vote_average']].to_dict(orient='records')

    try:
        # Generate mathematical coordinates locally on your computer instantly
        embeddings = model.encode(documents, convert_to_numpy=True).tolist()
        
        # Save directly to your local storage folder
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully indexed rows {i} to {min(i + batch_size, total_movies)} / {total_movies}")
        
    except Exception as e:
        print(f"Skipped batch starting at index {i} due to error: {e}")
        continue

end_time = time.time()
print("\n=======================================================")
print(f"SUCCESS: Local Vector Database built in {(end_time - start_time)/60:.2f} minutes!")
print("Saved permanently to 'chroma_storage/'")
print("=======================================================")
