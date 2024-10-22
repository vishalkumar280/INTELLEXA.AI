import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Index
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn
import nest_asyncio
import numpy as np

# Necessary for Colab or environments that don't support asyncio natively
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount the static directory to serve HTML files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("index1.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Load sentence transformer model (for embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Milvus (Assuming Milvus is running locally on default port 19530)
connections.connect("default", host="localhost", port="19530")

# Define the collection schema for storing vectors
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_model.get_sentence_embedding_dimension()),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
]

collection_name = "vector_collection"

# Create or load the collection in Milvus
try:
    collection = Collection(name=collection_name)
    logger.info(f"Collection '{collection_name}' already exists.")
except Exception as e:
    schema = CollectionSchema(fields, description="Text embedding collection")
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Collection '{collection_name}' created.")

# Create an index on the embedding field for faster search
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
Index(collection, "embedding", index_params)
logger.info(f"Index created on collection '{collection_name}'.")

# Load the collection for querying and inserting data
collection.load()

# Scrape Wikipedia content
def scrape_wikipedia(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page, status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all('p')
    content = "  ".join([para.get_text() for para in paragraphs])
    return content

# Embed and store the content in Milvus
def embed_and_store_text(content):
    sentences = content.split('.')
    embeddings = embedding_model.encode(sentences).tolist()
    entities = [embeddings, sentences]
    collection.insert(entities)
    logger.info(f"Inserted {len(sentences)} sentences with embeddings into collection.")

# Data model for the /load endpoint
class LoadDataRequest(BaseModel):
    url: str
class QueryData(BaseModel):
    query: str

# Corrected Function to generate text with Gemini API
def generate_text_with_gemini(prompt: str) -> str:
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    api_key = "AIzaSyBtfwcLyQ8kFpQ_Jo27gc5Ugh78m_YLptY"  # Replace with your actual API key
    full_url = f"{api_url}?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,  # Adjust the creativity level of the response
            "maxOutputTokens": 100  # Limit the number of tokens in the response
        }
    }

    response = requests.post(full_url, headers=headers, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        candidates = response_data.get('candidates', [])
        if candidates:
            return candidates[0].get('content', {}).get('parts', [{}])[0].get('text', 'No content generated')
        else:
            return "No content generated"
    else:
        raise Exception(f"Failed to generate content: {response.status_code} - {response.text}")

# Endpoint to load data from a Wikipedia URL into Milvus
@app.post("/load")
async def load_data(request: LoadDataRequest):
    try:
        # Scrape the Wikipedia content
        content = scrape_wikipedia(request.url)

        # Embed and store the content in Milvus
        embed_and_store_text(content)

        return {"message": "Data loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to query data and use Gemini for generating the final answer
# Endpoint to query data and use Gemini for generating the final answer
# Endpoint to query data and use Gemini for generating the final answer
@app.post("/query", response_model=dict)
async def query_data(request: QueryData):
    try:
        if collection.num_entities == 0:
            raise HTTPException(status_code=400, detail="No data loaded")

        query_embedding = embedding_model.encode([request.query]).tolist()[0]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 50}
        }
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["text"]
        )

        # Retrieve the most similar sentences and remove duplicates
        best_sentences = list({hit.entity.get("text") for hit in results[0]})

        # Combine best sentences into one string
        combined_best_sentences = " ".join(best_sentences)

        # Generate a response using Gemini
        gemini_response = generate_text_with_gemini(request.query).strip()

        logger.info(f"Best Sentences: {combined_best_sentences}")
        logger.info(f"Gemini Response: {gemini_response}")

        # Combine both best sentences and Gemini response
        final_response = f"{combined_best_sentences} {gemini_response}"

        # Return the combined response
        return {
            "combined_response": final_response
        }
    except Exception as e:
        logger.error(f"Failed to query data: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)