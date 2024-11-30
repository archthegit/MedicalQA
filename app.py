import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.embeddings = None

    def build_index(self, documents: List[str]):
        self.documents = documents
        self.embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        scores = np.dot(self.embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{"document": self.documents[i], "score": float(scores[i])} for i in top_indices]

class ResponseGeneration:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt: str, max_length: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class MedicalQASystem:
    def __init__(self):
        self.retriever = SimilarityMatching()
        self.generator = ResponseGeneration()

    def build_index(self, documents: List[str]):
        self.retriever.build_index(documents)

    def get_answer(self, query: str, top_k: int = 3):
        retrieved_docs = self.retriever.search(query, top_k)
        context = "\n".join([doc["document"] for doc in retrieved_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return {"answer": self.generator.generate_response(prompt), "sources": [doc["document"] for doc in retrieved_docs]}

qa_system = MedicalQASystem()

documents = [
    "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, and fatigue.",
    "Treatment for high blood pressure includes lifestyle changes and medication prescribed by doctors.",
    "Common causes of migraines include stress, certain foods, hormonal changes, and environmental factors.",
]
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
        return QueryResponse(answer=response["answer"], sources=response["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
