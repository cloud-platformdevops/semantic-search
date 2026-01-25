# Semantic Search using Langchain for RAG

This project demonstrates how to build a basic semantic search system using LangChain and Google Gemini embeddings. The process involves embedding document data(using weburl as a example), storing the embeddings in a vector store, and retrieving relevant results through semantic similarity search.

At query time, user input is embedded and compared against stored vectors to identify semantically relevant documents.

In a later stage, an LLM can be used to post-process or refine or summerize the retrieved data before presenting it to the user. The primary goal of this project is to showcase how semantic search works in practice.

# Prerequesites
Requires Python 3.10+
Using Gemini using Google AI studio: https://aistudio.google.com/welcome
you can find multiple LLM integration Libraries below link
https://docs.langchain.com/oss/python/integrations/providers/overview

# Concepts
### 1. Documents and Document Loaders
Langchain Document loaders give us one simple way to pull in information from many different soures like Web, PDF Slack, Notion, or Google Drive and turn it into LangChain’s Document format. This makes all the data easy to work with in the same way, no matter where it originally came from.
We can find more categories of source : https://docs.langchain.com/oss/python/integrations/document_loaders#by-category

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

### 4. Vector Store