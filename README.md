# Overview

<p>
    <a href="https://llm-pbe.github.io/document">
            <img alt="Build" src="https://img.shields.io/badge/1.0-document-orange">
    </a>
    <a href="https://arxiv.org/abs/2408.12787">
            <img alt="Build" src="https://img.shields.io/badge/arXiv-2408.12787-green">
    </a>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/1.12-PyTorch-orange">
    </a>
</p>

**LLM-PBE** is a toolkit to assess the data privacy of LLMs. The code is used for the [LLM-PBE](https://llm-pbe.github.io/home) [![arXiv](https://img.shields.io/badge/arXiv-2408.12787-green)](https://arxiv.org/abs/2408.12787) benchmark, which was selected as the :trophy: [Best Research Paper Nomination in VLDB 2024](https://llm-pbe.github.io/vldb2024_nomination_Qinbin.pdf).

## Getting Started
 

### Setup Environment

```shell
conda create -n llm-pbe python=3.10 -y
conda activate llm-pbe
# If you encounter the issue of 'kernel image' when running torch on GPU, try to install a proper torch with cuda.
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/microsoft/analysing_pii_leakage.git
pip install wandb accelerate
pip install -r requirements.txt
```



### Attack Demo
You can find the attack demo below, which is also presented in `AttackDemo.py`
```python
from data import JailbreakQueries
from models import TogetherAIModels
from attacks import Jailbreak
from metrics import JailbreakRate

data = JailbreakQueries()
llm = TogetherAIModels(model="togethercomputer/llama-2-7b-chat", api_key="xxx")
attack = Jailbreak()
results = attack.execute_attack(data, llm)
rate = JailbreakRate(results).compute_metric()
print("rate:", rate)
```

### Evaluate DP model metrics
```python
dp_evaluation = metrics.Evaluate(attack_dp_metrics, ground_truths=dataset.labels)
# Output results
print(f"Attack metrics on regular model: {evaluation}")
print(f"Attack metrics on DP model: {dp_evaluation}")
```

### Finetuning LLMs

Finetuning code is hosted separately with a different environment setup. Please refer to [Private Finetuning for LLMs (LLM-PFT)](https://github.com/jyhong836/llm-dp-finetune).

# Oliver and Tomas Documentation

## General Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--api` | Determines the API used for the model. Available API options shown below. |
| `--model` | Specifies the model to be used. |

## Available API Options

| Option | Description |
|--------|-------------|
| `ollama` | Uses the Ollama model running locally. |
| `open-webui` | Uses the OpenWebUI model. Requires API key and URL set in the environment variables `OPENWEBUI_KEY` and `OPENWEBUI_URL`. |
| `gpt` | Uses the ChatGPT model. Requires an OpenAI API key set in the environment variable `OPENAI_KEY`. |
| `hugging-face` | Uses models from Hugging Face. |
| `peft` | Uses the PeftCasualLM model. Requires `model`, `arch`, and `max_seq_len` arguments. |
| `claude` | Uses the ClaudeLLM model. |
| `meta-llama` | Uses the FinetunedCasualLM model. Requires `model`, `arch`, and `max_seq_len` arguments. |
| `together` | Option for the Together API. Requires API key set in the environment variables `TOGETHER_API_KEY`.|

## Running the Jailbreak Attack

Simple example of running the jailbreak script:

```bash
python -m scripts.jailbreak 
--api=ollama 
--model=llama3.2:1b
```

If you want to run the attack using intention analysis you can run with the `--intent_model` flag like this:

```bash
python -m scripts.jailbreak \
--api=ollama \
--model=qwen3:30b \
--intent_model=llama3.2:1b
```

## Running the Prompt Leakage Attack

Simple example of running the prompt leakage script:

```bash
python -m scripts.extract_prompt --api=ollama \
--data=blackfriday \
--num_test=10 \
--model=llama3.2:1b
```

## Running the Data Extraction Attack

Note that the attack can be run on models not trained on the dataset, but if the model has not trained on the dataset then the results might not be very useful. 

Simple example of running the data extraction attack for the enron dataset:

```bash
python -m attacks.DataExtraction.extract_enron_local \
--api=ollama \
--num_sample=10 \
--model=ollmoe-enron:latest
```

## Running the Membership Inference Attack

Note that the attack can be run on models not trained on the dataset, but if the model has not trained on the dataset then the results might not be very useful. 

This attack requires the usage of `llama.cpp` for calculating the perplexity. 

Example of running the membership inference attack without paths:

```bash
python -m attacks.MIA.run 
--metric=PPL 
--num_sample=1000 
--seed_test_set=123 
--model=<ollam-model-name> 
--data=enron 
--llama_cpp_path=<llama_cpp_path> 
--api=llama_cpp
```

Example of running the membership inference attack with paths:

```bash
python -m attacks.MIA.run 
--metric=PPL 
--num_sample=1000 
--seed_test_set=123 
--data=enron 
--model=/home/olitom/my-workspace/hf-models/olmoe-1B-7B-0125-Instruct-enron_Q4_K_M-gguf/olmoe-1B-7B-0125-Instruct-enron_Q4_K_M.gguf 
--llama_cpp_path=/home/olitom/my-workspace/llama.cpp/bin/llama-perplexity 
--api=llama_cpp
```