import requests

# Define Ollama server endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Define the model and prompt
data = {
    "model": "llama3",
    "prompt": "Summarize what GraphRAG is in 2 sentences.",
    "stream": False  # Use True for streaming responses
}

# Send the request
response = requests.post(OLLAMA_URL, json=data)

# Print the response
if response.status_code == 200:
    print("🧠 LLaMA 3 says:\n")
    print(response.json()["response"])
else:
    print(f"Error {response.status_code}: {response.text}")
