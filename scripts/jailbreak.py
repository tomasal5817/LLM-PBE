"""Jailbreak"""
#from attacks.Jailbreak.jailbreak import Jailbreak
from attacks import Jailbreak
from data import JailbreakQueries
from metrics import JailbreakRate
from models.togetherai import TogetherAIModels
from models.hf_models import HFModels
from models.chatgpt import ChatGPT
from models.open_webui import OpenWebUI
from models.togetherai import TogetherAIModels
import argparse
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from transformers import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--mulle', default=False, type=bool, help='Use Mulle API')
parser.add_argument('--model', default="llama3.2:1b", type=str, choices=[
    # models: https://platform.openai.com/docs/models
    # https://docs.together.ai/docs/inference-models
    # "EleutherAI/pythia-14m",
    # "EleutherAI/pythia-31m",
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m",
    # "EleutherAI/pythia-410m",
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-12b",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4",
    # together
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "llama3.2:1b",
    # not support system prompt
    # "mistralai/Mistral-7B-Instruct-v0.1",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # 
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "lmsys/vicuna-13b-v1.5-16k",
    "togethercomputer/falcon-7b-instruct",
    "togethercomputer/falcon-40b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "claude-2.1",
    ])

args = parser.parse_args()

print(f"== model: {args.model} ==")
if 'gpt' in args.model:
    api_key = os.getenv("OPENAI_KEY")
    llm = ChatGPT(api_key=api_key, model=args.model, max_attempts=30, max_tokens=2048)
# elif 'pythia' in args.model:
    # llm = HFModels(args.model=args.model, max_length=500)
elif 'claude' in args.model:
    from models.claude import ClaudeLLM
    llm = ClaudeLLM(model=args.model)
elif args.mulle:
    api_key = os.getenv("MULLE_KEY")
    base_url = os.getenv("MULLE_URL")
    url = f'{base_url}/api/chat/completions'
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'MULLE_KEY' is not set.")
    if not url:
        raise ValueError("Missing URL: Environment variable 'MULLE_URL' is not set.")
    llm = OpenWebUI(api_key=api_key, model=args.model, max_attempts=2, model_path=url)
else:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'TOGETHER_API_KEY' is not set.")
    llm = TogetherAIModels(api_key=api_key, model=args.model, max_attempts=2)

data = JailbreakQueries()
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)