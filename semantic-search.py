from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
import re

class SematicSearch():

    def __init__(self):
        print('init')
        PERSIST_DIR = "./chroma_db"
        self.client = genai.Client()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = Chroma(
            collection_name="handbook",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,  # Where to save data locally, remove if not necessary
        )
        print('done')

    def search(self, query):
        # Perform semantic search
        print("[INFO] Running semantic search")
        
        results = self.vector_store.similarity_search(query)

        # Display results
        print("\n[RESULTS]")
        for idx, doc in enumerate(results, start=1):
            content = doc.page_content.replace("\n", " ")
            # print(f"\nResult {idx}:\n{content}")   

            content = doc.page_content.replace("\\n", "\n") 
            # content = re.sub(r"\n(\d+)\s", r"\n\n**\1.** ", content)
            print(f"\nResult {idx}:\n{content}")    
         
semantic_search = SematicSearch()
semantic_search.search("what are the Private access options available?")