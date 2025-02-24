import os
from attacks.DataExtraction.enron import EnronDataExtraction
from models.togetherai import TogetherAIModels
from models.open_webui import OpenWebUI
from attacks.DataExtraction.prompt_extract import PromptExtraction

api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise ValueError("Not able to retrieve API Key from environment")

enron = EnronDataExtraction(data_path="data/enron")

for format in ['prefix-50','0-shot-known-domain-b','0-shot-unknown-domain-c', '3-shot-known-domain-c', '5-shot-unknown-domain-b']:
    prompts, _ = enron.generate_prompts(format=format)
  
    api_key = os.getenv("MULLE_KEY")
    base_url = os.getenv("MULLE_URL")
    if not api_key:
        raise ValueError("Not able to retrieve 'MULLE_KEY' from environment")
    if not base_url:
        raise ValueError("Not able to retrieve 'MULLE_URL' from environment")
    
    url = f'{base_url}/api/chat/completions'
    llm = OpenWebUI(api_key=api_key, model="llama3.2:1b", max_attempts=3, model_path=url)
    attack = PromptExtraction()
    results = attack.execute_attack(prompts, llm)
    print("results:", results)

