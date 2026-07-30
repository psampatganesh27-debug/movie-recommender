'''

import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(page_title="AI Movie Recommender", layout="centered")
st.title("🎬 Global AI Movie Vibe Recommender")
st.write("Search across 60,000+ movies with 100% stable offline database matching.")

@st.cache_resource
def load_assets():
    ai_client = genai.Client()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="chroma_storage")
    collection = chroma_client.get_collection(name="global_movies_recommender")
    return ai_client, embed_model, collection

ai_client, embed_model, collection = load_assets()

user_query = st.text_input(label="What kind of movie are you looking for?", placeholder="e.g., intense space survival or mind-bending heist thriller")

if st.button("Find Matches"):
    if user_query.strip() != "":
        with st.spinner("Searching through 60,000 movies..."):
            
            # Phase 1: Local Vector Database Query
            query_vector = embed_model.encode([user_query]).tolist()
            search_results = collection.query(query_embeddings=query_vector, n_results=4)
            
            context_block = ""
            
            # Phase 2: Pull the records straight from our offline metadata matrix
            for idx in range(len(search_results['ids'][0])):
                title = search_results['metadatas'][0][idx]['title']
                rating = search_results['metadatas'][0][idx]['vote_average']
                streaming_on = search_results['metadatas'][0][idx]['streaming_info']
                overview = search_results['documents'][0][idx]
                
                context_block += f"Movie Title: {title}\n"
                context_block += f"Rating: {rating}\n"
                context_block += f"Plot Summary: {overview}\n"
                context_block += f"Available On: {streaming_on}\n"
                context_block += "====================================\n\n"
            
            # Phase 3: Rigid Prompt Constraint Routing for Gemini
            system_instruction = (
                "You are an expert movie recommendation engine. Review the retrieved data blocks. "
                "For each movie, generate a clean Markdown header using '###' and the movie name. "
                "Provide a compelling 2-sentence explanation of why it fits the user's vibe request. "
                "Then, display a bold line indicating where it is currently streaming based on the 'Available On' data provided."
            )
            
            response = ai_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=f"User Request: {user_query}\n\nContext Block:\n{context_block}",
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )
            
            st.success("Top Recommendations Found:")
            st.markdown(response.text)
'''

import os
import glob
import zipfile
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

load_dotenv()

# Option A: Unpack single chroma_storage.zip if present
if not os.path.exists("chroma_storage") and os.path.exists("chroma_storage.zip"):
    st.info("Unpacking vector database index...")
    with zipfile.ZipFile("chroma_storage.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Option B: Reassemble and unpack split parts from chroma_storage_parts if present
elif not os.path.exists("chroma_storage") and os.path.exists("chroma_storage_parts"):
    st.info("Reassembling and extracting vector database index... Please wait a moment.")
    parts = sorted(glob.glob("chroma_storage_parts/part_*.bin"))
    combined_zip = "temp_chroma_storage.zip"
    
    with open(combined_zip, "wb") as outfile:
        for part in parts:
            with open(part, "rb") as infile:
                outfile.write(infile.read())
                
    with zipfile.ZipFile(combined_zip, "r") as zip_ref:
        zip_ref.extractall(".")
        
    os.remove(combined_zip)

st.set_page_config(page_title="AI Movie Recommender", layout="centered")
st.title("🎬 Global AI Movie Vibe Recommender")
st.write("Search across 60,000+ movies powered by local embeddings and AI intelligence.")

@st.cache_resource
def load_assets():
    # Automatically reads GEMINI_API_KEY from environment or Streamlit Cloud Secrets
    ai_client = genai.Client()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="chroma_storage")
    
    # Safely get or create the collection
    collection = chroma_client.get_or_create_collection(name="global_movies_recommender")
    return ai_client, embed_model, collection

ai_client, embed_model, collection = load_assets()

user_query = st.text_input(
    label="What kind of movie are you looking for?", 
    placeholder="e.g., intense space survival or mind-bending heist thriller"
)

if st.button("Find Matches"):
    if user_query.strip() != "":
        with st.spinner("Searching database catalog..."):
            
            # Phase 1: Vector Database Query
            query_vector = embed_model.encode([user_query]).tolist()
            search_results = collection.query(query_embeddings=query_vector, n_results=4)
            
            context_block = ""
            
            # Verify if records were retrieved
            if search_results and search_results['ids'] and len(search_results['ids'][0]) > 0:
                for idx in range(len(search_results['ids'][0])):
                    title = search_results['metadatas'][0][idx]['title']
                    rating = search_results['metadatas'][0][idx]['vote_average']
                    overview = search_results['documents'][0][idx]
                    
                    context_block += f"Movie Title: {title}\n"
                    context_block += f"Database Rating: {rating} / 10\n"
                    context_block += f"Plot Summary: {overview}\n"
                    context_block += "-------------------\n\n"
                
                system_instruction = (
                    "You are an expert movie recommendation engine. Review the retrieved movies provided in the context.\n\n"
                    "For each movie:\n"
                    "1. Create a clean Markdown header using '###' and the movie name.\n"
                    "2. Provide a bold sub-headline indicating its rating, e.g., '**Rating:** [Insert Rating Value] / 10'.\n"
                    "3. Provide a compelling 2-sentence explanation of why it fits the user's vibe query.\n"
                    "4. Create a dedicated bold line called '**Where to Watch:**'. Use your internal knowledge base "
                    "to list the major streaming platforms where this movie is widely available (e.g., Netflix, Prime Video, Disney+, Apple TV, JioCinema, etc.)."
                )
                
                response = ai_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=f"User Request: {user_query}\n\nRetrieved Matches:\n{context_block}",
                    config=types.GenerateContentConfig(system_instruction=system_instruction)
                )
                
                st.success("Top Recommendations Found:")
                st.markdown(response.text)
            else:
                st.error("Vector database is empty. Please ensure 'chroma_storage' or 'chroma_storage_parts' is uploaded to GitHub.")