import os
from data import JailbreakQueries
from attacks import Jailbreak
from metrics import JailbreakRate
from models.open_webui import OpenWebUI

api_key = os.getenv("MULLE_KEY")
base_url = os.getenv("MULLE_URL")

if not api_key:
    raise ValueError("Not able to retrieve API Key from environment")

elif not base_url:
    raise ValueError("Not able to retrieve URL address from environment")

url = f'{base_url}/api/chat/completions'
data = JailbreakQueries()
llm = OpenWebUI(api_key=api_key, model="llama3.2:1b", max_attempts=3, model_path=url)
print("model:", llm.model)
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
# print("rate:", rate)
llm.logger.info('Rate: %s', rate)
