import os
import pickle  # Import to save/load vectors
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import torch
import time
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarityMatching:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", vector_store_file: str = "vector_store"):
        self.documents = []
        self.embeddings = None
        self.knowledge_vector_database = None
        self.embedding_model_name = embedding_model_name
        self.vector_store_file = vector_store_file
        self.load_vector_store()  # Attempt to load vector store if available

    def build_index(self, documents):
        self.documents = documents
        if self.knowledge_vector_database is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Found device: {device}")

            embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                multi_process=True,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("model loaded successfully")

            start_time = time.time()
            print("Creating the vector database...")
            # tqdm

            self.knowledge_vector_database = FAISS.from_documents(
                documents, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            end_time = time.time()

            elapsed_time = (end_time - start_time) / 60
            print(f"Time taken to create the vector database: {elapsed_time} minutes")
            
            # Save vector store after creating it
            self.save_vector_store()
        else:
            print("Vector store loaded successfully from file.")

    def search(self, query: str, top_k: int = 3):
        results = self.knowledge_vector_database.similarity_search(query, k=top_k)
        # print(results)
        return [
            {
                "document_id": result.metadata.get("document_id"),
                "source": result.metadata.get("source"),
                "url": result.metadata.get("url"),
                "content": result.page_content,
                "index": i
            }
            for i, result in enumerate(results)
        ]
    def save_vector_store(self):
        # Save vector store to a file for later reuse
        with open(self.vector_store_file, "wb") as f:
            pickle.dump(self.knowledge_vector_database, f)
        print(f"Vector store saved successfully to {self.vector_store_file}")

    def load_vector_store(self):
        # Load vector store if it exists
        if os.path.exists(self.vector_store_file):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                multi_process=True,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.knowledge_vector_database = FAISS.load_local(self.vector_store_file, embedding_model,allow_dangerous_deserialization=True)
            print(f"Vector store loaded successfully from {self.vector_store_file}")
        else:
            print(f"No vector store file found. A new one will be created.")



class ResponseGeneration:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt: str, max_length: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class DocumentPreparer:
    def __init__(self, folder_path=None):
        self.df = None
        self.docs = []
        if folder_path:
            self.load_data(folder_path)

    def load_data(self, folder_path):
        try:
            self.df = pd.read_csv(folder_path)
            print(f"Data loaded successfully from: {folder_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {folder_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def prepare_docs(self):
        if self.df is None:
            print("Error: DataFrame is not loaded. Please load the data first.")
            return

        self.docs = []

        for _, row in tqdm(self.df.iterrows(), desc="Preparing documents", total=len(self.df)):
            page_content = f"Question: {row['question']} \nAnswer: {row['answer']}"
            metadata = {
                "document_id": row["document_id"],
                "source": row["source"],
                "url": row["url"],
            }
            doc = Document(page_content=page_content, metadata=metadata)
            self.docs.append(doc)

        print(f"Total documents prepared: {len(self.docs)}")

    def chunk_documents(self):
        if not self.docs:
            print("Error: No documents available to chunk. Please prepare the documents first.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=768,
            chunk_overlap=128,
        )
        doc_snippets = text_splitter.split_documents(self.docs)
        self.docs = doc_snippets
        
        print(f"Total document chunks prepared: {len(self.docs)}")

    def get_docs(self):
        return self.docs


class MedicalQASystem:
    def __init__(self):
        self.retriever = SimilarityMatching()
        self.generator = ResponseGeneration()

    def build_index(self, documents):
        self.retriever.build_index(documents)

    def get_answer(self, query: str, top_k: int = 3):
        retrieved_docs = self.retriever.search(query, top_k)
        context = "\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = self.generator.generate_response(prompt)
        returnobj = {
            "answer": answer,
            "sources": [
                {
                    "document": doc["document_id"],
                    "sourcedoc": doc["source"],
                    "url": doc["url"]
                } for doc in retrieved_docs
            ]
        }
        print(returnobj)
        return returnobj

document_preparer = DocumentPreparer("medquad_data.csv")
document_preparer.load_data("medquad_data.csv")
document_preparer.prepare_docs()
document_preparer.chunk_documents()  # Chunk the documents after preparation

documents = document_preparer.get_docs()
qa_system = MedicalQASystem()
qa_system.build_index(documents)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/")
def root():
    return {"message": "Welcome to the MedInquire API"}

@app.post("/get-answer", response_model=QueryResponse)
def get_answer(request: QueryRequest):
    try:
        response = qa_system.get_answer(request.query, request.top_k)
        sources = [
            f"Document ID: {doc['document']}, Source: {doc['sourcedoc']}, URL: {doc['url']}"
            for doc in response["sources"]
        ]

        return QueryResponse(answer=response["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
