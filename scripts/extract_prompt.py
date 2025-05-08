"""Prompt Leakage"""
from data.prompt_leakage import PromptLeakageSysPrompts
# from models.togetherai import TogetherAIModels
from attacks.PromptLeakage.prompt_leakage import PromptLeakage
from models.ft_clm import FinetunedCasualLM, PeftCasualLM
from models.hf_models import HFModels
from models.chatgpt import ChatGPT
from models.open_webui import OpenWebUI
from models.togetherai import TogetherAIModels
from models.ollama import Ollama
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import torch
import wandb
from transformers import set_seed


parser = argparse.ArgumentParser()
parser.add_argument('--mulle', default=False, type=bool, help='Use Mulle API')
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat', type=str)
parser.add_argument('--peft', default='lora', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--num_test', default=10, type=int, help='num of sys prompts to extract.')
parser.add_argument('--max_seq_len', default=1024, type=int)
parser.add_argument('--data', default='blackfriday', type=str, choices=['blackfriday', 'GPTs', 'blackfriday/Academic', 'blackfriday/Business', 'blackfriday/Creative', 'blackfriday/Game', 'blackfriday/Job-Hunting', 'blackfriday/Marketing', 'blackfriday/Productivity-&-life-style', 'blackfriday/Programming'])
parser.add_argument('--model', default="meta-llama/Llama-2-7b-chat-hf", type=str)
parser.add_argument('--api', default='together', type=str, help='Api endpoint', choices=['peft', 'gpt', 'hugging-face', 'claude', 'open-webui', 'ollama', 'meta-llama', 'together'])
parser.add_argument('--defense', default=None, type=str, choices=[
    "no-repeat",
    "top-secret",
    "ignore-ignore-inst",
    "no-ignore",
    "eaten",
])
args = parser.parse_args()

set_seed(args.seed)

args.run_name = args.model.split("/")[-1]+f"_s{args.seed}"
if args.defense is not None:
    args.run_name += f"_d-{args.defense}"
print(f"run name: {args.run_name}")

wandb.init(project='LLM-PBE', name=args.run_name, config=vars(args))

out_dir = f'results/prompt_leakage/{args.data}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

data = PromptLeakageSysPrompts(category=args.data)
sys_prompts = data.random_select(args.num_test, seed=args.seed)

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
elif args.api == 'ollama':
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

attack = PromptLeakage()
# try:
results = attack.execute_attack(sys_prompts, llm)

# for key, value in results.items():
#    wandb.log({print(f"{key}: {value}")}, commit=False)

fname = os.path.join(out_dir, args.run_name  + '.pth')
torch.save(results, fname)

attack_scores = defaultdict(list)
for attack_prompt_name, gen_prompts in results.items():
    match_scores = attack.compute_scores(sys_prompts, gen_prompts)
    attack_scores['attack'].append(attack_prompt_name)
    attack_scores['defense'].append(args.defense)
    attack_scores['match_scores'].append(np.mean(match_scores))
    
    wandb.log({f'{attack_prompt_name}_score': np.mean(match_scores)}, commit=False)

wandb.log({'max_score': np.max(attack_scores['match_scores'])}, commit=True)

df = pd.DataFrame(attack_scores)
print(df)
