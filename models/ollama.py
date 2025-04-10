import time
from models.LLMBase import LLMBase
from ollama import show, chat, ChatResponse

class Ollama(LLMBase):
    def __init__(self,api_key=None, model='llama3.2', model_path=None, max_attempts=3):
        super().__init__(api_key=api_key, model_path=model_path)
        try:
            show(model)
        except Exception as e:
            raise Exception(f"Ollama or Ollama-model is not available {e}")
        self.model=model
        self.max_attempts=max_attempts
        self.delay_seconds=3
        self.tokenizer=None

    def load_model(self):
        pass
        
    def query_local_model(self, query, messages=None):
        n_attempt = 0
        while n_attempt < self.max_attempts:
            try:
                response: ChatResponse = chat(model=self.model, messages=[
                    {
                        'role': 'user',
                        'content': query,
                    },
                ])
                return response['message']['content']
            except Exception as e:
                print(f"An error occurred: {e}")
                n_attempt += 1
                time.sleep(self.delay_seconds)
        if n_attempt == self.max_attempts:
            raise Exception("Max number of attempts reached")
            exit(1)
        return response['message']['content']
