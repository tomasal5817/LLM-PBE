import os
from models.open_webui import OpenWebUI

api_key = os.getenv("MULLE_KEY")
base_url = os.getenv("MULLE_URL")

if not api_key:
    raise ValueError("Not able to retrieve API Key from environment")
elif not base_url:
    raise ValueError("Not able to retrieve URL address from environment")

url = f'{base_url}/api/chat/completions'
query = 'Hello, tell me a joke about llamas.'
llm = OpenWebUI(api_key=api_key, model="llama3.2:1b",
                max_attempts=3, model_path=url)
response = llm.query_remote_model(query)
print(f'Response: {response}')
