from models.ft_clm import PeftCasualLM, FinetunedCasualLM
from models.llama_cpp import Llama_cpp 
from attacks.MIA.member_inference import MemberInferenceAttack, MIAMetric
from transformers import BertForMaskedLM, BertTokenizer
import argparse
import wandb
import os
import numpy as np

def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)


parser = argparse.ArgumentParser()
parser.add_argument('--metric', default='perplexity', type=str)
parser.add_argument('--num_sample', default=3000, type=int, help='use -1 to include all samples')
parser.add_argument('--data', default='echr', type=str, choices=['echr', 'enron'])
parser.add_argument('--model', default='LLM-PBE/echr-llama2-7b-chat-undefended', type=str)
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat-hf', type=str)
parser.add_argument('--peft', default='lora', type=str)
parser.add_argument('--max_seq_len', default=1024, type=int)
parser.add_argument('--n_neighbor', default=50, type=int, help='num of neighbors in neighbor attack')

parser.add_argument('--seed_test_set', default=None, type=int, help='whether to seed the selection of test set.')

#parser.add_argument('--model_path', defaul=None, type=str, help='Path to GGUF model')
parser.add_argument('--llama_cpp_path', default=None, type=str, help='Path to llama.cpp')
parser.add_argument('--api', default='together', type=str, help='Api endpoint', choices=['peft', 'gpt', 'hugging-face', 'claude', 'open-webui', 'ollama', 'meta-llama', 'together', 'llama_cpp'])

args = parser.parse_args()

args.run_name = f"{args.metric}_{args.num_sample}"
if args.max_seq_len != 1024:
    args.run_name += f"_len{args.max_seq_len}"
if args.data != 'echr':
    args.run_name += f"_{args.data}"
if args.n_neighbor != 50:
    args.run_name += f"_nn{args.n_neighbor}"
args.result_dir = os.path.join("./results/", f"{args.model}_{args.peft}")
make_if_not_exist(args.result_dir)
cache_file = os.path.join(args.result_dir, args.run_name)

wandb.init(project='LLM-PBE')

metric = MIAMetric[args.metric]

if args.data == 'echr':
    from data.echr import EchrDataset
    ds = EchrDataset(data_path="data/echr", pseudonymize=False)
elif args.data == 'enron':
    from data.enron import EnronDataset
    ds = EnronDataset(data_path="data/enron", pseudonymize=False)
else:
    raise NotImplementedError(f"data: {args.data}")
train_set = ds.train_set()
if args.num_sample > 0 and args.num_sample < len(train_set):
    train_set = train_set.select(range(args.num_sample))
test_set = ds.test_set()
if args.num_sample > 0 and args.num_sample < len(test_set):
    if args.seed_test_set is None:
        idxs = range(args.num_sample)
    else:
        idxs = np.random.RandomState(args.seed_test_set).choice(len(test_set), args.num_sample, replace=False)
    test_set = test_set.select(idxs)

if args.api == 'llama_cpp': # Run test localy on llama.cpp
    if args.llama_cpp_pathNone is not None and args.model is not None:
        llm = Llama_cpp(model=args.model, llama_cpp=args.llama_cpp)
    else:
        raise ValueError("You must specify --llama_cpp_path and --model")
elif args.peft == 'none':
    llm = FinetunedCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
else:
    # Replace api_key with your own API key
    # llm = PeftCasualLM(model_path='LLM-PBE/echr-llama2-7b-undefended', arch='meta-llama/Llama-2-7b-hf')
    # llm = PeftCasualLM(model_path='LLM-PBE/echr-llama2-7b-chat-dp8', arch='meta-llama/Llama-2-7b-chat-hf')
    llm = PeftCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
if metric in (MIAMetric.REFER, MIAMetric.LIRA, MIAMetric.NEIGHBOR):
    ref_llm = FinetunedCasualLM(model_path=args.arch, arch=args.arch, max_seq_len=args.max_seq_len)
    ref_llm._lm.eval()
else:
    ref_llm = None

attack = MemberInferenceAttack(metric=metric, ref_model=ref_llm, n_neighbor=args.n_neighbor)
print(f"Results cache => {cache_file}")
results = attack.execute(llm, train_set, test_set, cache_file=cache_file, resume=True)
score_dict = attack.evaluate(results)
print("results:", score_dict)
wandb.log(score_dict)
wandb.finish()