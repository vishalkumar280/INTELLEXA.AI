"""

**LYFNGO**
"""

!pip install fastapi uvicorn beautifulsoup4 requests sentence-transformers faiss-cpu transformers pyngrok nest-asyncio -q
!pip install fastapi[all] -q

#import necessary libraries
import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
from pyngrok import ngrok
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

# Apply nest_asyncio for Jupyter/Colab environments
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the SentenceTransformer model for creating embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the Hugging Face generative model pipeline (e.g., GPT-2 or similar)
generative_model = pipeline("text-generation", model="gpt2")


# Test Hugging Face text generation
test_model = pipeline("text-generation", model="gpt2")
test_output = test_model("This is a test.", max_length=100)
print(test_output)


# In-memory FAISS index
faiss_index = None
stored_sentences = []
SCRAPED_TEXT_FILE = "/content/scraped_text.txt"  # File to store scraped text

# Define request models for FastAPI
class LoadDataRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# Function to scrape Wikipedia content
def extract_wikipedia_data(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page, status code: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])

    # Save the scraped text to a file
    with open(SCRAPED_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    return text

# Function to load text from file (if available)
def load_text_from_file() -> str:
    if os.path.exists(SCRAPED_TEXT_FILE):
        with open(SCRAPED_TEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return None

# Function to embed text and load it into FAISS
def embed_and_store_text(text: str):
    global faiss_index, stored_sentences

    # Split text into sentences
    sentences = text.split(". ")
    embeddings = embedding_model.encode(sentences)

    # Initialize FAISS index with cosine similarity
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)  # Normalize to use cosine similarity
    faiss_index.add(np.array(embeddings).astype("float32"))

    stored_sentences = sentences

# FastAPI endpoint to load data from Wikipedia and store embeddings in FAISS
@app.post("/load")
def load_data(request: LoadDataRequest):
    try:
        # Step 1: Scrape Wikipedia data (always fetch new data)
        content = extract_wikipedia_data(request.url)

        # Step 2: Embed the text and store embeddings in FAISS
        embed_and_store_text(content)

        return {"message": "Data loaded successfully", "total_sentences": len(stored_sentences)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint to query the vector database and generate a response using a generative AI model
@app.post("/query")
def query_data(request: QueryRequest):
    try:
        if faiss_index is None or faiss_index.ntotal == 0:
            raise HTTPException(status_code=400, detail="FAISS index is not loaded with data")

        # Step 1: Embed the user query
        query_embedding = embedding_model.encode([request.query])[0].astype("float32")
        print("Query Embedding Shape:", query_embedding.shape)

        # Step 2: Search for the top 3 most relevant sentences in FAISS
        distances, indices = faiss_index.search(np.array([query_embedding]), k=3)
        print("Distances:", distances)
        print("Indices:", indices)

        best_sentences = [stored_sentences[i] for i in indices[0]]
        print("Best Matching Sentences:", best_sentences)

        # Step 3: Combine the retrieved sentences into a prompt for the generative AI model
        # Limit to the first 2 sentences to reduce the length of the prompt
        prompt = " ".join(best_sentences[:2]) + f" Based on this, answer the question: {request.query}"
        print("Prompt:", prompt)

        # Step 4: Generate a response using the generative AI model with max_new_tokens and truncation
        generated_response = generative_model(
            prompt,
            max_new_tokens=50,  # Adjust this value for how long the response should be
            truncation=True,  # Truncate if the input is too long
        )[0]['generated_text']
        print("Generated Response:", generated_response)

        # Return the AI-generated answer along with the best match sentences
        return {
            "query": request.query,
            "best_match_sentences": best_sentences,
            "generated_answer": generated_response
        }
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# Authenticate Ngrok with your authtoken
ngrok_auth_token = '2mt1V0s487izXCkFecJdYcO5ucZ_2C6GTWT3XsGEYAP1F7MRM'  # Replace with your actual ngrok authtoken
!ngrok authtoken {ngrok_auth_token}

# Start ngrok tunnel to expose the local FastAPI app to the internet
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

# Serve the static background image
@app.get("/background")
async def get_background():
    return FileResponse("/content/Background_template.jpg")  # Adjust the path accordingly
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>INTELLEXA</title>

    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* General Page Styling */
        body {
            font-family: 'Roboto', sans-serif;
            background: url('/background') no-repeat center center fixed;
            background-size: cover;
            color: #ffffff;
            text-align: center;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin: 0;
        }

        .content-wrapper {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: flex-end;
            text-align: right;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            position: relative;
            top: 150px; /* Moves the content downward */
            left: 150px; /* Moves the content rightward */
        }

        /* Input container for aligning text box and button */
        .input-container {
            display: flex;
            align-items: center;
            margin: 10px 0;
            width: 100%;
            justify-content: flex-end;
        }

        input[type="text"] {
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border: 2px solid white;
            width: 100%;
            max-width: 400px;
            font-size: 1.1em;
            background-color: rgba(255, 255, 255, 0.05);
            color: #ffffff;
            outline: none;
        }

        input[type="text"]::placeholder {
            color: rgba(240, 240, 240, 0.8);
        }

        button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 15px 30px;
            cursor: pointer;
            font-weight: normal;
            font-size: 1em;
            margin-left: 10px; /* Space between the input field and the button */
            transition: background-color 0.3s, transform 0.3s;
        }

        button:hover {
            background-color: #0056b3;
        }

        button:active {
            transform: scale(0.95);
        }

        #loadResult, #questionResult {
            margin-top: 20px;
            font-size: 1.1em;
            color: #ffffff;
            text-align: right;
            font-family: 'Roboto', sans-serif;
        }

        #answerResult {
            background-color: rgba(0, 0, 0, 0.6);
            color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: left;
            line-height: 1.5;
            max-width: 100%;
            width: 600px;
            max-height: 200px; /* Limit the height of the answer area */
            overflow-y: auto; /* Enable vertical scroll */
        }

        #answerResult b {
            color: #ffffff;
        }

        @media screen and (max-width: 768px) {
            input[type="text"] {
                width: 90%;
            }
            #answerResult {
                width: 90%;
            }

            .content-wrapper {
                top: 100px;
                left: 50px;
            }
        }

        /* Styling the scroll bar */
        #answerResult::-webkit-scrollbar {
            width: 8px;
        }

        #answerResult::-webkit-scrollbar-thumb {
            background-color: rgba(255, 255, 255, 0.3); /* Customize the scrollbar */
            border-radius: 10px;
        }
    </style>
    <script>
        let isDataLoaded = false;  // Track if data has been loaded

        async function loadData() {
            const url = document.getElementById('inputField').value;
            const loadResponse = await fetch('/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            if (loadResponse.ok) {
                const loadData = await loadResponse.json();
                document.getElementById('loadResult').textContent = "DATA IS LOADED SUCCESSFULLY";
                isDataLoaded = true;  // Set flag indicating data is loaded
            } else {
                document.getElementById('loadResult').textContent = "Failed to load data.";
            }
        }

        async function queryData() {
            if (!isDataLoaded) {
                alert("Please load data before querying!");
                return;
            }

            const query = document.getElementById('inputFieldQuery').value;
            const queryResponse = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            if (queryResponse.ok) {
                const queryData = await queryResponse.json();
                document.getElementById('questionResult').textContent = `Question: ${queryData.query}`;

                // Display the best-matching sentences
                let answersHtml = `Answer:<br>`;
                queryData.best_match_sentences.forEach(sentence => {
                    answersHtml += `<b>${sentence}</b><br>`;
                });
                document.getElementById('answerResult').innerHTML = answersHtml;
            } else {
                document.getElementById('questionResult').textContent = "Error querying data.";
            }
        }
    </script>
</head>
<body>
    <div class="content-wrapper">
        <div class="input-container">
            <input type="text" id="inputField" placeholder="Enter a Wikipedia URL">
            <button onclick="loadData()">Load Data</button>
        </div>

        <div id="loadResult"></div>

        <div class="input-container">
            <input type="text" id="inputFieldQuery" placeholder="Ask a question">
            <button onclick="queryData()">Ask AI</button>
        </div>

        <div id="questionResult"></div>
        <div id="answerResult"></div>
    </div>
</body>
</html>
"""
# FastAPI route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
def get_html():
    return HTMLResponse(content=html_code)

# Start the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)