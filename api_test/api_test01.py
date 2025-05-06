import requests

API_URL = "https://huggingface.co/uer/gpt2-chinese-cluecorpussmall"

def get_response(text):
    response = requests.post(API_URL, json={"inputs": text})
    print(response.json())

if __name__ == "__main__":
    get_response("你好")
