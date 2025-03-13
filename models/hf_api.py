import os
import requests
import time
from models.LLMBase import LLMBase

class HFAPI(LLMBase):
    '''
    Initial test class for Hugging Face API, mainly for testing pythia-1.4b model
    Consider setting API key in constructor or not.
    '''
    def __init__(self, api_key=None, model_path=None, model='pythia-1.4b-v0', x_use_cache="true", x_wait_for_model="false", max_attempts = 3):
        super().__init__(api_key=api_key, model_path=model_path)
        self.api_url = f'https://api-inference.huggingface.co/models/EleutherAI/{model}'
        if api_key is None:
            self.api_key = os.getenv("HF_API_KEY")
            if not self.api_key:
                raise ValueError("Not able to retrieve API Key from environment")
        self.model = model
        self.max_attempts = max_attempts
        self.delay_seconds = 3
        self.tokenizer = None
        self.x_use_cache = x_use_cache
        self.x_wait_for_model = x_wait_for_model
   
    def load_model(self):
        pass
        
    def query_remote_model(self, prompt, messages=None):
        n_attempt = 0
        if messages is None:
            messages = {"inputs": prompt}
        while n_attempt < self.max_attempts:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "x-use-cache": self.x_use_cache,
                    "x-wait-for-model": self.x_wait_for_model
                }
                data = messages

                response = requests.post(self.api_url, headers=headers, json=data)
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

        output = response.json()
        generated_text = output[0]['generated_text']
        return generated_text