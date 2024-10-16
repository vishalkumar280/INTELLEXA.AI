INTELLEXA.AI Project
Overview
INTELLEXA.AI is a Python-based project designed to dynamically extract and query knowledge from Wikipedia pages using a combination of web scraping, machine learning, and natural language processing (NLP) techniques. The project integrates a vector database for efficient data storage and retrieval, and uses a generative AI model for answering user queries based on the extracted data.
Key Features
•	Web Scraping: Automatically extracts data from a single Wikipedia page.
•	Vector Database Storage: Uses FAISS to store and index vector representations of the extracted text.
•	Generative AI: Leverages a pre-trained GPT-2 model to generate answers based on the stored data.
•	FastAPI Endpoints: Provides two main endpoints for loading and querying data.
•	Ngrok Integration: Exposes the FastAPI application to the internet for external access and testing.
Problem Statement
The project addresses the following key requirements:
•	Extract data from a single Wikipedia page.
•	Load the extracted data into a vector database for efficient querying.
•	Utilize a generative AI model to answer questions based on the loaded data.
•	Implement two FastAPI endpoints:
o	/load: Loads Wikipedia data into the vector database.
o	/query: Queries the generative AI model to generate answers from the stored data.
Solution Workflow
1.	Web Scraping:
o	Uses BeautifulSoup and Python's requests library to scrape text from Wikipedia.
o	Extracts content from HTML <p> tags and processes it into a usable format.
2.	Vector Database Storage:
o	Converts extracted text into vector embeddings using SentenceTransformers (all-MiniLM-L6-v2).
o	Stores these embeddings in FAISS for efficient querying using cosine similarity.
3.	Generative AI:
o	A pre-trained GPT-2 model is used to generate context-aware answers based on the relevant data retrieved from FAISS.
o	Combines user queries with stored context to generate appropriate responses.
4.	FastAPI Endpoints:
o	/load: Scrapes and loads Wikipedia data into FAISS.
o	/query: Handles user queries and retrieves relevant information from the vector database to generate answers.
5.	Ngrok:
o	Exposes the FastAPI app to the public via a secure Ngrok tunnel for testing and external access.
Modules
•	Web Scraping: Extracts text from Wikipedia.
•	Embedding & Storage: Embeds text using SentenceTransformers and stores it in FAISS.
•	Generative AI: GPT-2 model for question answering.
•	FastAPI: RESTful API for interaction.
•	Ngrok: Public access to the API for testing purposes.
Requirements
•	Python 3.8+
•	Libraries: beautifulsoup4, requests, sentence-transformers, faiss-cpu, transformers, fastapi, uvicorn, pyngrok
•	Ngrok for tunneling: https://ngrok.com/
Installation
1.	Clone the repository:
git clone https://github.com/your-repo/INTELLEXA.AI.git
2.	Install the required dependencies:
pip install -r requirements.txt
3.	Set up Ngrok:
ngrok authtoken YOUR_NGROK_AUTH_TOKEN
Usage
1.	Run the FastAPI server:
uvicorn main:app --reload
2.	Expose the API to the internet:
ngrok http 8000
3.	Load Wikipedia data:
o	Endpoint: /load
o	Provide the Wikipedia URL to scrape and store in FAISS.
4.	Query data:
o	Endpoint: /query
o	Ask questions based on the previously loaded Wikipedia data.
Example
•	Load Wikipedia data:
POST /load
{
"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"
}
•	Query the data:
POST /query
{
  				 "question": "What is Python?"
}
Project Structure
•	main.py: The core FastAPI application.
•	scraping.py: Handles web scraping from Wikipedia.
•	embedding.py: Converts text into vector embeddings.
•	query.py: Handles querying from FAISS and generating answers via GPT-2.

