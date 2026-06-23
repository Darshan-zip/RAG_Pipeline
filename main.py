import ollama
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()  
pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

with open("sample.txt", "r") as file:
    dataset = file.readlines()
dataset=" ".join(dataset)

dense_index_name = "dense-vectors-final"

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

dense_index = pc.Index(dense_index_name)

records = []
sents=dataset.split(".")
for i,sent in enumerate(sents):
  if not sent:
    continue
  records.append({
    "_id":f"rec{i}",
    "chunk_text":sent
  })
NAMESPACE_NAME="final-namespace"
dense_index.upsert_records(records=records, namespace=NAMESPACE_NAME)


sparse_index_name = "sparse-vectors-final"

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
sparse_index = pc.Index(sparse_index_name)
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
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        grade = response['message']['content'].strip().capitalize()
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
    # 1. Initial Retrieval
    retrieved_data = retrieve(input_query)
    
    # 2. Corrective Step: Grading
    relevant_docs = grade_documents(input_query, retrieved_data)
    
    
    print(f"\n[CRAG] {len(relevant_docs)} relevant documents found.")

    # 3. Final Generation
    context_text = '\n'.join([f"- {doc['document']['chunk_text']}" for doc in relevant_docs])
    
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information. If the given pieces of context does not contain any information related to the user query, then just say that "I do not have enough information to answer this question":
    {context_text}'''

    LANGUAGE_MODEL='llama3'

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )
    
    print('\nChatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
user_input=''
while user_input!="x":
    user_input=input("\n\nQuery:")
    RAG(user_input)