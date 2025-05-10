import requests

API_URL = "https://router.huggingface.co/hf-inference/models/Qwen/Qwen3-235B-A22B/v1/chat/completions"
headers = {
    "Authorization": "Bearer hf_xxxxxx",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "Qwen/Qwen3-235B-A22B"
})

print(response["choices"][0]["message"])