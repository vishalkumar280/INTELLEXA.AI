INTELLEXA.AI

OVERVIEW:

INTELLEXA.AI is a Python-based solution designed to dynamically extract, store, and query knowledge from Wikipedia using machine learning and natural language processing (NLP) techniques. It incorporates a generative AI model for answering user queries based on the extracted data.

Key Features
Web Scraping: Extracts content from Wikipedia pages using BeautifulSoup.

Vector Database: Stores data in a FAISS-based vector database for efficient retrieval.

Generative AI: Uses GPT-2 to generate answers based on stored data.

FastAPI Endpoints: Provides RESTful endpoints for loading and querying data.

Ngrok Integration: Exposes the API for external access and testing.

WORKING:

Scrape and extract text from a Wikipedia page.

Convert text into vector embeddings and store them in FAISS.

Use a pre-trained GPT-2 model to generate answers from the stored data.

Access the application via FastAPI, with public exposure through Ngrok.


For more detailed information, refer to the project documentation in this repository.
