import os
import pickle
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import openai
import base64

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
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", vector_store_file="vector_store"):
        self.documents = []
        self.knowledge_vector_database = None
        self.embedding_model_name = embedding_model_name
        self.vector_store_file = vector_store_file
        self.load_vector_store()

    def build_index(self, documents):
        self.documents = documents
        if self.knowledge_vector_database is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.knowledge_vector_database = FAISS.from_documents(
                documents, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            self.save_vector_store()
        else:
            print("Vector store loaded successfully from file.")

    def search(self, query: str, top_k: int = 3):
        results = self.knowledge_vector_database.similarity_search(query, k=top_k)
        return [
            {
                "document_id": result.metadata.get("document_id"),
                "source": result.metadata.get("source"),
                "url": result.metadata.get("url"),
                "content": result.page_content,
            }
            for result in results
        ]

    def save_vector_store(self):
        with open(self.vector_store_file, "wb") as f:
            pickle.dump(self.knowledge_vector_database, f)

    def load_vector_store(self):
        if os.path.exists(self.vector_store_file):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.knowledge_vector_database = FAISS.load_local(
                self.vector_store_file, embedding_model, allow_dangerous_deserialization=True
            )
        else:
            print(f"No vector store file found. A new one will be created.")

class ResponseGeneration:
    def __init__(self, openai_key):
        """
        Initializes the ResponseGeneration class for OpenAI models.
        :param openai_key: OpenAI API key
        """
        if not openai_key:
            raise ValueError("OpenAI API key is required.")
        openai.api_key = openai_key

    def generate_response(self, prompt: str, model="gpt-4", max_length: int = 256):
        """
        Generates a response using OpenAI's ChatCompletion API.
        :param prompt: The input prompt for response generation
        :param model: The OpenAI model to use
        :param max_length: Maximum length of the response
        :return: Generated response as a string
        """
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.7
            )

            # print("==== RESPONSE ===")
            # print(response)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error while generating response: {e}")
            return "I'm sorry, I couldn't generate a response. Please try again later."


class DocumentPreparer:
    def __init__(self, folder_path=None):
        self.df = None
        self.docs = []
        if folder_path:
            self.load_data(folder_path)

    def load_data(self, folder_path):
        try:
            self.df = pd.read_csv(folder_path)
        except Exception as e:
            print(f"Error: {e}")

    def prepare_docs(self):
        for _, row in tqdm(self.df.iterrows(), desc="Preparing documents", total=len(self.df)):
            doc = Document(
                page_content=f"Question: {row['question']} \nAnswer: {row['answer']}",
                metadata={
                    "document_id": row["document_id"],
                    "source": row["source"],
                    "url": row["url"],
                },
            )
            self.docs.append(doc)

    def chunk_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=768, chunk_overlap=128)
        self.docs = text_splitter.split_documents(self.docs)

    def get_docs(self):
        return self.docs

class MedicalQASystem:
    def __init__(self, openai_key, vector_store_file="vector_store"):
        self.retriever = SimilarityMatching(vector_store_file=vector_store_file)
        self.generator = ResponseGeneration(openai_key=openai_key)
        self.previous_reconstructed_history = ""
        if "previous_reconstructed_query.txt" in os.listdir():
            self.previous_reconstructed_history = open("previous_reconstructed_query.txt","r").read()
        self.image_context = ""
        if "previous_image_history.txt" in os.listdir():
            self.image_context = open("previous_image_history.txt","r").read()

    def build_index(self, documents):
        if not self.retriever.knowledge_vector_database:
            self.retriever.build_index(documents)
        else:
            print("Vector store already exists. Skipping index building.")

    def clean_context(self, context: str) -> str:
        paragraphs = list(set(context.split("\n")))
        cleaned_context = "\n".join(para.strip() for para in paragraphs if para.strip())
        return " ".join(cleaned_context.split()[:200])

    def query_enhancer(self, query: str):
        print(self.previous_reconstructed_history)
        if self.previous_reconstructed_history == "":
            prompt = (f"You are a professional query reconstructor. Based on the query provided below, construct a enhanced"
                      f" query that a large language model would be able to comprehend better and" 
                      f" if the query refers to an image, change the query to refer to the image context"
                      f"Query: {query}\nReconstructed Query:\n"
            
            )
            reconstructed_query = self.generator.generate_response(prompt).strip("")
            f = open("previous_reconstructed_history.txt","w")
            f.write(reconstructed_query)
            f.close()
            self.previous_reconstructed_history = reconstructed_query
            return reconstructed_query
        else:
            prompt = (f"You are a professional query reconstructor. Based on the query provided and the"
                      f" previous reconstructed query with context included below, construct a enhanced"
                      f" query that a large language model would be able to comprehend better and"
                      f" if the query refers to an image, change the query to refer to the image context"
                      f"Query: {query}\n previous reconstructed query: {self.previous_reconstructed_history} Reconstructed Query:\n"
            
            )
            reconstructed_query = self.generator.generate_response(prompt).strip("")
            print("Writing to file")
            f = open("previous_reconstructed_history.txt","w")
            f.write(reconstructed_query)
            f.close()
            self.previous_reconstructed_history = reconstructed_query
            return reconstructed_query


    def get_answer(self, query: str, top_k: int = 3):
        reconstructed_query = self.query_enhancer(query)
        print(reconstructed_query)
        retrieved_docs = self.retriever.search(reconstructed_query, top_k)
        context = self.clean_context("\n".join([doc["content"] for doc in retrieved_docs]))
        if self.image_context == "":
            prompt = (
            f"You are a professional medical assistant. Based on the context below, provide a detailed and "
            f"professional response to the user's question. Avoid repetition. "
            f"If the context is insufficient, inform the user politely.\n\nContext:\n{context}\n\n"
            f"Question: {reconstructed_query}\nAnswer:"
        )
        else:
            prompt = (
            f"You are a professional medical assistant, it's alright if you can't view images, don't refelct"
            f" that in your response and instead use the Image Context"
            f" Based on the context below, provide a detailed and "
            f"professional response to the user's question. Avoid repetition. "
            f"If the context is insufficient, inform the user politely.\n\nContext:\n{context}\n\n"
            f"Question: {reconstructed_query}\n Image Context: {self.image_context} \nAnswer:"
        )
        answer = self.generator.generate_response(prompt)
        return {
            "answer": answer.strip(),
            "sources": retrieved_docs,
        }

# document_preparer = DocumentPreparer("medquad_data.csv")
# document_preparer.load_data("medquad_data.csv")
# document_preparer.prepare_docs()
# document_preparer.chunk_documents()

# documents = document_preparer.get_docs()
# qa_system = MedicalQASystem(openai_key="")
# qa_system.build_index(documents)



document_preparer = None
if not os.path.exists("vector_store"):
    document_preparer = DocumentPreparer("medquad_data.csv")
    document_preparer.load_data("medquad_data.csv")
    document_preparer.prepare_docs()
    document_preparer.chunk_documents()


documents = document_preparer.get_docs() if document_preparer else []
qa_system = MedicalQASystem(openai_key="api_key_here", vector_store_file="vector_store")

if documents:
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

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Process the uploaded file
    contents = await file.read()
    base64_str = base64.b64encode(contents).decode("utf-8")
    img_type = file.filename.split(".")[-1]
    print("===== IMAGE TYPE =====")
    print(img_type)
    image_description = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":"Describe the below image"},
                    {
                        "type":"image_url",
                        "image_url": {"url":f"data:image/{img_type};base64,{base64_str}"},
                    },
                ]
            }
        ]
    ).choices[0].message.content
    f = open("previous_image_history.txt","w")
    f.write(image_description)
    # Perform image processing or file handling here
    return {"message": f"File {file.filename} received and processed successfully."}


@app.post("/get-answer", response_model=QueryResponse)
def get_answer(request: QueryRequest):
    try:
        response = qa_system.get_answer(request.query, request.top_k)
        sources = [
            f"{doc['source']} (Document ID: {doc['document_id']}) - {doc['url']}"
            for doc in response["sources"]
        ]
        return QueryResponse(answer=response["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
