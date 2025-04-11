"""Jailbreak"""
#from attacks.Jailbreak.jailbreak import Jailbreak
from attacks import Jailbreak
from data import JailbreakQueries
from metrics import JailbreakRate
from models.togetherai import TogetherAIModels
from models.hf_models import HFModels
from models.chatgpt import ChatGPT
from models.open_webui import OpenWebUI
from models.ollama import Ollama
from models.togetherai import TogetherAIModels
from models.ft_clm import PeftCasualLM, FinetunedCasualLM
import argparse
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from transformers import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat', type=str)
parser.add_argument('--max_seq_len', default=1024, type=int)
parser.add_argument('--peft', default='none', type=str)
parser.add_argument('--model', default="llama3.2:1b", type=str)
parser.add_argument('--api', default='together', type=str, help='Chose api, regarding to model chose and.', choices=['peft', 'gpt', 'hugging-face', 'claude', 'open-webui', 'ollama', 'meta-llama', 'together'])

args = parser.parse_args()

print(f"== model: {args.model} ==")
if args.api == 'peft':
    llm = PeftCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
elif args.api == 'gpt':
    api_key = os.getenv("OPENAI_KEY")
    llm = ChatGPT(api_key=api_key, model=args.model, max_attempts=30, max_tokens=2048)
elif args.api == 'hugging-face':
    llm = HFModels(model_name=args.model)
elif args.api == 'claude':
    from models.claude import ClaudeLLM
    llm = ClaudeLLM(model=args.model)
elif args.api == 'open-webui':
    api_key = os.getenv("MULLE_KEY")
    base_url = os.getenv("MULLE_URL")
    url = f'{base_url}/api/chat/completions'
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'MULLE_KEY' is not set.")
    if not url:
        raise ValueError("Missing URL: Environment variable 'MULLE_URL' is not set.")
    llm = OpenWebUI(api_key=api_key, model=args.model, max_attempts=2, model_path=url)
elif args.api == ollama:
    llm = Ollama(model=args.model, max_attempts=2)
elif args.api == 'meta-llama':
     llm = FinetunedCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
elif args.api == 'together':
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'TOGETHER_API_KEY' is not set.")
    llm = TogetherAIModels(api_key=api_key, model=args.model, max_attempts=2)
else:
    raise ValueError('No valid api endpoint')
    
data = JailbreakQueries(["Joe Biden", "Barrack Obama", "Kamala Harris", "Nancy Pelosi", "Hillary Clinton", "Chuck Schumer", "Elizabeth Warren", "Bernie Sanders", "Alexandria Ocasio-Cortez", "Mitch McConnell"])
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)
