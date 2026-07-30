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
st.write("Search across 60,000+ movies powered by local embeddings and AI intelligence.")

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
        with st.spinner("Searching database catalog..."):
            
            # Phase 1: Local Vector Database Query
            query_vector = embed_model.encode([user_query]).tolist()
            search_results = collection.query(query_embeddings=query_vector, n_results=4)
            
            context_block = ""
            
            context_block = ""
            
            # Phase 2: Pull clean data straight from the local vector database (with ratings re-added!)
            for idx in range(len(search_results['ids'][0])):
                title = search_results['metadatas'][0][idx]['title']
                rating = search_results['metadatas'][0][idx]['vote_average']
                overview = search_results['documents'][0][idx]
                
                # We append the rating directly into the context payload string for Gemini to read
                context_block += f"Movie Title: {title}\n"
                context_block += f"Database Rating: {rating} / 10\n"
                context_block += f"Plot Summary: {overview}\n"
                context_block += "-------------------\n\n"
            
            # Phase 3: Tell Gemini exactly how to display the rating metrics
            system_instruction = (
                "You are an expert movie recommendation engine. Review the retrieved movies provided in the context.\n\n"
                "For each movie:\n"
                "1. Create a clean Markdown header using '###' and the movie name.\n"
                "2. Provide a bold sub-headline indicating its rating, e.g., '**Rating:** [Insert Rating Value] / 10'.\n"
                "3. Provide a compelling 2-sentence explanation of why it fits the user's vibe query.\n"
                "4. Create a dedicated bold line called '**Where to Watch:**'. Use your internal knowledge base "
                "to list the major streaming platforms where this movie is widely available (e.g., Netflix, Prime Video, Disney+, Apple TV, JioCinema, etc.)."
            )
            
            # Phase 4: Conversational Synthesis
            response = ai_client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=f"User Request: {user_query}\n\nRetrieved Matches:\n{context_block}",
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )
            
            st.success("Top Recommendations Found:")
            st.markdown(response.text)
'''
