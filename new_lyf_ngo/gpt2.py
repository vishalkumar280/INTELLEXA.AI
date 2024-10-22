
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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

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

# Load the GPT-2 model and tokenizer
gpt2_model_name = 'gpt2'  # You can specify other variants like 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)

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

# Function to generate text with GPT-2
def generate_text_with_gpt2(prompt: str, max_length: int = 100) -> str:
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Create attention mask: 1s for real tokens and 0s for padding tokens
    # Check if the tokenizer has a pad token
    if tokenizer.pad_token_id is not None:
        attention_mask = (inputs != tokenizer.pad_token_id).to(torch.long)  # Convert to long type
    else:
        attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)  # All ones if no padding

    outputs = gpt2_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated




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

# Endpoint to query data and use GPT-2 for generating the final answer
import json

# Endpoint to query data and use GPT-2 for generating the final answer
# Endpoint to query data and use GPT-2 for generating the final answer
# Endpoint to query data and use GPT-2 for generating the final answer
# Endpoint to query data and use GPT-2 for generating the final answer
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

        # Log the best sentences to see what was retrieved
        logger.info(f"Best Sentences: {best_sentences}")

        # Generate a response using GPT-2
        gpt2_response = generate_text_with_gpt2(request.query).strip()

        # Ensure GPT-2 response does not repeat the query
        gpt2_response = gpt2_response.replace(request.query, "").strip()

        logger.info(f"GPT-2 Response: {gpt2_response}")

        # Return only the GPT-2 response
        return {
            "gpt2_response": gpt2_response
        }
    except Exception as e:
        logger.error(f"Failed to query data: {e}")
        raise HTTPException(status_code=500, detail=str(e))






if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)