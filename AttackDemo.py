import os
from data import JailbreakQueries
from models import TogetherAIModels
from attacks import Jailbreak
from metrics import JailbreakRate

api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise ValueError("Not able to retrieve API Key from environment")

data = JailbreakQueries()
llm = TogetherAIModels(api_key=api_key, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", max_attempts=2)
print("model:", llm.model)
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)
