# INTELLEXA.AI

INTELLEXA.AI is a powerful application designed to extract, process, and query Wikipedia data using state-of-the-art machine learning techniques. This project integrates web scraping, vector embeddings, generative AI models, and an interactive web UI to create a fully functional knowledge retrieval system.

Features:

Web Scraping with BeautifulSoup: Extract data dynamically from Wikipedia pages.

Sentence Embeddings: Use the sentence-transformers/all-MiniLM-L6-v2 model to convert text into vector embeddings.

Vector Database with Milvus: Store the embeddings in Milvus, a highly scalable vector database.

Generative AI Model: Utilize the Gemini 1.5 Flash model for generating responses to user queries.

FastAPI: Provide two endpoints for loading Wikipedia data into the database and querying the generative AI model.

Interactive Web Page: Engage with the system using a web interface that supports querying and provides real-time results.

GPT-2 Integration: The project also includes GPT-2 for text generation, which can be accessed via Swagger UI(additional quering model).

Project Structure:

1. Web Scraping:

The project utilizes BeautifulSoup for web scraping to extract Wikipedia content.

2. Embedding and Vector Database:

Embeddings: The extracted data is transformed into vectors using the all-MiniLM-L6-v2 model (SentenceTransformer).

Milvus Database: The embeddings are stored in the Milvus vector database, which runs via Docker.

3. Generative AI and Querying:

The Gemini 1.5 Flash model is used to query the vector database and retrieve relevant information based on user input.

A FastAPI endpoint (/query) processes the user input, retrieves data from Milvus, and sends a prompt to the generative AI model.

4. FastAPI Endpoints:

/load: Loads data from a Wikipedia URL into the vector database.

/query: Accepts user queries and returns answers based on the data in the vector database.

5. Web Interface:

The web interface allows users to load Wikipedia data and query the system through an intuitive UI. The interface features:

A background image.
Inputs for loading Wikipedia URLs and submitting queries.
Display of the results directly on the page.

How It Works:


Load Wikipedia Data: Input a Wikipedia URL on the web interface, and the system will scrape the content, convert it to vector embeddings, and store it in Milvus.

Query System: Enter a query, and the system will retrieve relevant information from the vector database and generate a response using Gemini 1.5 Flash.

Addtional query system: The GPT-2 model can also be tested through the Swagger UI but is not directly connected to the web UI.

Technologies Used:

BeautifulSoup: Web scraping.

Sentence-Transformers: all-MiniLM-L6-v2 for embedding generation.

Milvus: Vector database for efficient storage and retrieval of embeddings.

FastAPI: Backend framework for serving the API.

Docker: For containerizing Milvus.

Gemini 1.5 Flash: Generative AI model for producing text responses.

GPT-2: Additional generative model, accessible via Swagger UI.

HTML/CSS: For building the interactive web page.

Conclusion:

INTELLEXA.AI showcases how to integrate web scraping, vector databases, and generative AI to create a system capable of querying and generating answers based on dynamic web data. The combination of FastAPI, Milvus, and advanced AI models makes this project both flexible and powerful for various knowledge retrieval use cases.
