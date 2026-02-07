# Semantic Search using Langchain for RAG

This project demonstrates how to build a basic semantic search system using LangChain and Google Gemini embeddings. The process involves embedding document data(using weburl as a example), storing the embeddings in a vector store, and retrieving relevant results through semantic similarity search.

At query time, user input is embedded and compared against stored vectors to identify semantically relevant documents.

In a later stage, an LLM can be used to post-process or refine or summerize the retrieved data before presenting it to the user. The primary goal of this project is to showcase how semantic search works in practice.

# Prerequesites
Requires Python 3.10+

Using Gemini using Google AI studio: https://aistudio.google.com/welcome

you can find multiple LLM integration Libraries below link
https://docs.langchain.com/oss/python/integrations/providers/overview

# Installation
Install the required packages using pip:
```
pip install -r requirements.txt
```
Create a .env file in the root directory of the project and add the following environment variables:
```
GOOGLE_API_KEY=your_google_api_key
```

# Concepts
### 1. Documents and Document Loaders
Langchain Document loaders give us one simple way to pull in information from many different soures like Web, PDF Slack, Notion, or Google Drive and turn it into LangChain’s Document format. This makes all the data easy to work with in the same way, no matter where it originally came from.

We can find more categories of document loaders (like pdf, slack, notion etc) : https://docs.langchain.com/oss/python/integrations/document_loaders#by-category

### 2. Text Splitters
Text splitters take a long document and divide it into smaller, manageable chunks that can be retrieved independently and fit within a model’s context window limits.
Text is already arranged in layers—paragraphs, sentences, and words. Using this structure makes it easier to split a document into smaller pieces that still feel natural, keep the meaning intact, and adjust to whatever level of detail we need.

For most use cases, RecursiveCharacterTextSplitter should work
```
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
```
- separaters: List of separators to try in order. You can customize (e.g., split by "." for sentences).
- chunk_size: The maximum size of a chunk, where size is determined by the length_function.
- chunk_overlap: Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
- add_start_index: character index where each split Document starts within the initial Document is preserved as metadata attribute “start_index”.
- length_function: Function determining the chunk size.
- is_separator_regex: Whether the separator list (defaulting to ["\n\n", "\n", " ", ""]) should be interpreted as regex.
- keep_separator: If True, keeps the separator at the end of each chunk.

### 3. Embedding
An embedding is a way for a computer to understand meaning, not just words.

In simple terms, embeddings turn text (like words, sentences, or documents) into numbers that capture their meaning and relationships. These numbers are stored as a list (called a vector). Texts with similar meanings end up with similar numbers.

An embedding library doesn’t create intelligence by itself. Instead, it uses a pretrained embedding model to convert text into numbers that represent meaning. In this case, we are using Google Gemini embedding model.

When we pass input text to the embedding model, it cleans and convertes the text into tokens and pass it to the embedding model. The embedding model returns a vector representation of the text. 

### 4. Vector Store
A vector store/database is a special type of database designed to store, search, and compare embeddings.

Instead of saving text like traditional databases, a vector store saves vectors (lists of numbers) that represent the meaning of text, images, or other data.

Also, it can be used to simpler search for similar documents, images, or other data based on their embeddings. we can use vector store.similarity_search(query) to search for similar documents based on their embeddings.

Example:
```
vector store.similarity_search(query, k)
```


### 5. Retriever
A Retriever is a Runnable wrapper around a VectorStore in Langchain. It adapts the VectorStore into LangChain’s execution model. This makes it easy to use the vector store in a LangChain chain or pipeline, Async & batch execution, and streaming.

For simpler search, we can use vector store directly without using retriver. 
The Current project uses vector store directly without using retriver.

### 6. Similarity Search
Similarity search is a process of finding the most similar documents to a query based on their embeddings.

### 7. LLM
LLM (Large Language Model) is a type of AI model that can understand and generate human-like text based on the input it receives.


## More Info
https://docs.langchain.com/oss/python/langchain/knowledge-base