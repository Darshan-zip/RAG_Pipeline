import os
from ollama import Client
ollama_api_key = os.getenv("OLLAMA_API_KEY")
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {ollama_api_key}"}
)

print(client.list())