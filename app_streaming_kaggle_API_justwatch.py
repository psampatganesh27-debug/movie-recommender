import os
import requests
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# 1. System Setup
load_dotenv()
TMDB_TOKEN = os.getenv("TMDB_READ_TOKEN")

st.set_page_config(page_title="AI Movie Recommender", layout="centered")
st.title("🎬 Global AI Movie Vibe Recommender")
st.write("Search for any movie vibe using local vector embeddings and live streaming lookups.")

@st.cache_resource
def load_assets():
    ai_client = genai.Client()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="chroma_storage")
    collection = chroma_client.get_collection(name="global_movies_recommender")
    return ai_client, embed_model, collection

ai_client, embed_model, collection = load_assets()

# 2. TMDB Multi-Region Data API Layer
def get_streaming_platforms(movie_id, movie_title=""):
    if not TMDB_TOKEN:
        return "Streaming configuration token missing.", "#"
    
    try:
        clean_id = str(int(float(movie_id)))
    except Exception:
        clean_id = str(movie_id)
        
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_TOKEN}"
    }
    
    url = f"https://api.themoviedb.org/3/movie/{clean_id}/watch/providers"
    response = requests.get(url, headers=headers, timeout=5)
    
    # Auto-Correction Subsystem for Mismatched Row IDs
    if response.status_code == 404 and movie_title != "":
        search_url = "https://api.themoviedb.org/3/search/movie"
        try:
            search_res = requests.get(search_url, headers=headers, params={"query": movie_title}, timeout=5)
            if search_res.status_code == 200:
                results_array = search_res.json().get("results", [])
                if results_array:
                    clean_id = str(results_array[0]["id"])
                    url = f"https://api.themoviedb.org/3/movie/{clean_id}/watch/providers"
                    response = requests.get(url, headers=headers, timeout=5)
        except Exception:
            pass

    try:
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", {})
            
            # 1. India Subscription Check
            in_res = results.get("IN", {})
            if in_res.get("flatrate"):
                providers = [p["provider_name"] for p in in_res["flatrate"]]
                return f"Streaming in India on: {', '.join(providers)}", in_res.get("link", "#")
            
            # 2. International Subscription Fallback
            us_res = results.get("US", {})
            if us_res.get("flatrate"):
                providers = [p["provider_name"] for p in us_res["flatrate"]]
                return f"Streaming internationally on: {', '.join(providers)}", us_res.get("link", "#")
            
            # 3. Trans-regional Rent/Buy Fallback
            for region in ["IN", "US"]:
                if results.get(region, {}).get("link"):
                    return "Available to Rent/Buy or In Theaters", results[region]["link"]
            
            if results:
                first_reg = list(results.keys())[0]
                if results[first_reg].get("link"):
                    return "Available Internationally (Check options link)", results[first_reg]["link"]
            
            return "Not Streaming on Subscriptions Currently", "#"
    except Exception:
        pass
        
    return "Streaming Info Unavailable", "#"

# 3. Frontend Search Execution Layer
user_query = st.text_input(label="What kind of movie are you looking for?", placeholder="e.g., sci-fi psychological thriller")

if st.button("Find Matches"):
    if user_query.strip() != "":
        with st.spinner("Searching through 60,000 movies..."):
            
            # Phase 1: Local Vector Search
            query_vector = embed_model.encode([user_query]).tolist()
            search_results = collection.query(query_embeddings=query_vector, n_results=4)
            
            # Phase 2: Structural Data Loop & Formatting
            context_block = ""
            raw_ui_fallback_data = []
            
            for idx in range(len(search_results['ids'][0])):
                movie_id = search_results['ids'][0][idx]
                title = search_results['metadatas'][0][idx]['title']
                rating = search_results['metadatas'][0][idx]['vote_average']
                overview = search_results['documents'][0][idx]
                
                streaming_on, info_url = get_streaming_platforms(movie_id, movie_title=title)
                
                # Build explicit clean string boundaries using native newline configurations
                context_block += f"Movie Title: {title}\n"
                context_block += f"Database Rating: {rating}\n"
                context_block += f"Plot Overview: {overview}\n"
                context_block += f"Live Streaming: {streaming_on}\n"
                context_block += f"Watch Reference URL: {info_url}\n"
                context_block += "====================================\n\n"
                
                raw_ui_fallback_data.append({
                    "title": title,
                    "rating": rating,
                    "streaming": streaming_on,
                    "url": info_url
                })
            
            # Phase 3: Rigid Prompt Constraint Routing
            system_instruction = (
                "You are an expert movie recommendation engine. Review the retrieved movies. "
                "For each movie, generate a clean Markdown header using '###' and the movie name. "
                "Write a concise 2-sentence explanation explaining why this matches the query. "
                "Then, create a dedicated bold line called '**Where to Watch:**'. "
                "CRITICAL: Look at the 'Watch Reference URL' field in the context. If it contains a link "
                "starting with http, you MUST output a clickable markdown link matching the text: "
                "[Click Here for Official Watch Options](paste_the_exact_url_here). Do not change the URL."
            )
            
            try:
                response = ai_client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=f"User Request: {user_query}\n\nContext Dataset:\n{context_block}",
                    config=types.GenerateContentConfig(system_instruction=system_instruction)
                )
                
                st.success("Top Tailored Matches:")
                st.markdown(response.text)
                
            except Exception as gemini_error:
                # If the cloud API fails or skips parsing, fallback to rendering directly from the database matching layout
                st.warning("Displaying direct local database matches due to API layout restrictions:")
                for item in raw_ui_fallback_data:
                    st.subheader(item["title"])
                    st.write(f"Rating: {item['rating']}")
                    st.write(item["streaming"])
                    if item["url"] != "#":
                        st.markdown(f"[Click Here for Watch Options]({item['url']})")