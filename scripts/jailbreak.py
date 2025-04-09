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
from models.ft_clm import PeftCasualLM, FinetunedCasualLM
import argparse
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from transformers import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--mulle', default=False, type=bool, help='Use Mulle API')
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat', type=str)
parser.add_argument('--max_seq_len', default=1024, type=int)
parser.add_argument('--peft', default='none', type=str)
parser.add_argument('--model', default="llama3.2:1b", type=str)

args = parser.parse_args()

print(f"== model: {args.model} ==")
if args.peft != 'none':
    llm = PeftCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
elif 'gpt' in args.model:
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
elif 'meta-llama' in args.model:
     llm = FinetunedCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
else:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'TOGETHER_API_KEY' is not set.")
    llm = TogetherAIModels(api_key=api_key, model=args.model, max_attempts=2)

data = JailbreakQueries(["Joe Biden", "Barrack Obama", "Kamala Harris", "Nancy Pelosi", "Hillary Clinton", "Chuck Schumer", "Elizabeth Warren", "Bernie Sanders", "Alexandria Ocasio-Cortez", "Mitch McConnell"])
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)