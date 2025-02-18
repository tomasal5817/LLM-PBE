import requests
import time
from copy import deepcopy
import json
import logging
from datetime import datetime
from models.LLMBase import LLMBase

class OpenWebUI(LLMBase):
    def __init__(self, api_key, model_path='http://localhost:3000/api/chat/completions', model = "llama3.1", max_attempts = 3):
        super().__init__(api_key=api_key, model_path=model_path)
        self.model = model
        self.max_attempts = max_attempts
        self.delay_seconds = 3
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=f'logs/app_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log', filemode='w', level=logging.INFO) 

    def load_model(self):
        pass
        
    def query_remote_model(self, prompt, messages=None):
        n_attempt = 0
        if messages is None:
            messages = [{'role': 'user', 'content': prompt}]
        while n_attempt < self.max_attempts:
            try:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    'model': self.model,
                    'messages': messages
                    #'files': [{'type': 'file', 'id': 'test.txt'}]
                }
                response = requests.post(self.model_path, headers=headers, json=payload)
                if not response:
                    print('Empty response!')
            except Exception as e:
                # Catch any exception that might occur and print an error message
                print(f"An error occurred: {e}")
                n_attempt += 1
                time.sleep(self.delay_seconds)
            else:
                break
        if n_attempt == self.max_attempts:
            raise Exception("Max number of attempts reached")
            exit(1)
        
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        self.logger.info('Prompt: %s', prompt)
        self.logger.info('Response: %s', content)
        
        return content