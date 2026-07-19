"""
    Things to do:
      1. see the activity i.e., hallucination percentage, evaluvation
      2. providing citations for each response
    Future improvements:
      1. make the chunks cache more useful by adding semantic search
"""
import ollama
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from ollama import Client
load_dotenv()  
pinecone_api_key = os.getenv("PINECONE_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")
#TO maintain citation backed result do this: create a new database which automatically creates new entries whenever we try to enter data into the vector database.


pc = Pinecone(api_key=pinecone_api_key)

NAMESPACE_NAME="final-namespace"

dense_index_name = "dense-vectors-final"
dense_index = pc.Index(dense_index_name)
sparse_index_name = "sparse-vectors-final"
sparse_index = pc.Index(sparse_index_name)

LLM_NAME="ministral-3:14b"


import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)
import hashlib
import json

def serialize_docs(docs):
    return [doc.document["chunk_text"] for doc in docs]
  
def make_cache_key(prefix: str, text: str):
    normalized = text.strip().lower()
    return f"{prefix}:{hashlib.sha256(normalized.encode()).hexdigest()}"

def get_cached_response(query: str):
    key = make_cache_key("resp", query)
    cached = redis_client.get(key)
    return cached if cached else None
  
def set_cached_response(query: str, response: str):
    key = make_cache_key("resp", query)
    redis_client.set(key, response, ex=3600 * 24)
    
def get_cached_chunks(query: str):
    key = make_cache_key("retr", query)
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None


def set_cached_chunks(query: str, chunks: list):
    key = make_cache_key("retr", query)
    redis_client.set(key, json.dumps(chunks), ex=3600 * 24)
  
def ollama_chat(prompt):
    

    client = Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {ollama_api_key}"
        }
    )

    response = client.chat(
        model=LLM_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.message.content

def upsert():
  with open("sample.txt", "r") as file:
      dataset = file.readlines()
  dataset=" ".join(dataset)


  if not pc.has_index(dense_index_name):
      pc.create_index_for_model(
          name=dense_index_name,
          cloud="aws",
          region="us-east-1",
          embed={
              "model":"llama-text-embed-v2",
              "field_map":{"text": "chunk_text"}
          }
      )

  

  records = []
  sents=dataset.split(".")
  for i,sent in enumerate(sents):
    if not sent:
      continue
    records.append({
      "_id":f"rec{i}",
      "chunk_text":sent
    })

  dense_index.upsert_records(records=records, namespace=NAMESPACE_NAME)




  if not pc.has_index(sparse_index_name):
      pc.create_index_for_model(
          name=sparse_index_name,
          cloud="aws",
          region="us-east-1",
          embed={
              "model":"pinecone-sparse-english-v0",
              "field_map":{"text": "chunk_text"}
          }
      )
  
  sparse_index.upsert_records(records=records, namespace=NAMESPACE_NAME)



def merge_chunks(h1, h2):
    """Get the unique hits from two search results and return them as single array of {'_id', 'chunk_text'} dicts, printing each dict on a new line."""
    # Deduplicate by _id
    deduped_hits = {hit['id']: hit for hit in h1['result']['hits'] + h2['result']['hits']}.values()
    # Sort by _score descending
    sorted_hits = sorted(deduped_hits, key=lambda x: x['score'], reverse=True)
    # Transform to format for reranking
    result = [{'id': hit['id'], 'chunk_text': hit['fields']['chunk_text']} for hit in sorted_hits]
    return result
  

def grade_documents(query, documents):
    """Grade the retrieved documents and return only the relevant ones."""
    if not documents:
        return []

    graded_docs = []
    grading_prompt = f"""Grade the following document based on its relevance to the query. 
    Respond only with a single word: 'Relevant' or 'Irrelevant'.

    Query: {query}
    Document: {documents[0]['document']['chunk_text']}
    """
    
    # For simplicity in this implementation, we grade documents one by one
    # In a production system, you could batch this or use a specialized model
    for doc in documents:
        prompt = f"""Grade the following document based on its relevance to the query. 
        Respond only with a single word: 'Relevant' or 'Irrelevant'.

        Query: {query}
        Document: {doc['document']['chunk_text']}
        """
        response = ollama_chat(prompt)
        grade = response.strip().capitalize()
        if 'Relevant' in grade:
            graded_docs.append(doc)
    
    return graded_docs




def retrieve(query):
    dense_results = dense_index.search(
        namespace=NAMESPACE_NAME,
        top_k=40,
        inputs={
            "text": query
        }
    )

    sparse_results = sparse_index.search(
        namespace=NAMESPACE_NAME,
        top_k=40,
        inputs={
            "text": query
        }
    )
    merged_results = merge_chunks(sparse_results, dense_results)

    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=merged_results,
        rank_fields=["chunk_text"],
        top_n=10,
        return_documents=True,
        parameters={
            "truncate": "END"
        }
    )

    return result.data


  

def RAG(input_query):
    # 1. Check final response cache
    cached_response = get_cached_response(input_query)
    if cached_response:
        print("⚡ Cache HIT (response)")
        print(cached_response)
        return cached_response

    # 2. Check retrieval cache
    cached_chunks = get_cached_chunks(input_query)

    if cached_chunks:
        print("⚡ Cache HIT (chunks)")

        # IMPORTANT: convert cached chunks into same format as fresh path
        relevant_docs = [
            {"document": {"chunk_text": text}}
            for text in cached_chunks
        ]

    else:
        print("❌ Cache MISS")

        retrieved_data = retrieve(input_query)

        # cache ONLY raw texts
        cached_texts = serialize_docs(retrieved_data)
        set_cached_chunks(input_query, cached_texts)

        # grading still done only on fresh retrieval
        relevant_docs = grade_documents(input_query, retrieved_data)

        print(f"\n[CRAG] {len(relevant_docs)} relevant documents found.")

    # 3. Build context (UNIFIED FORMAT)
    context_text = "\n".join(
        [f"- {doc['document']['chunk_text']}" for doc in relevant_docs]
    )
    
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information. If the given pieces of context does not contain any information related to the user query, then just say that "I do not have enough information to answer this question":
    {context_text}'''



    client = Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {ollama_api_key}"
        }
    )

    stream = client.chat(
        model=LLM_NAME,
        messages=[
            
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True   
    )
    response=""
    print('\nChatbot response:')
    for chunk in stream:
        response+=f"{chunk['message']['content']}"
        print(chunk['message']['content'], end='', flush=True)
    
    set_cached_response(input_query, response)
        

user_input=''
while user_input!="x":
    user_input=input("\n\nQuery:")
    RAG(user_input)