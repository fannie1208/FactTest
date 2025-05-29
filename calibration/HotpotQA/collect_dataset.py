from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import numpy as np
import os



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def format_question(input_data):
    
    context_ls = []
    for single_context in input_data['context']:
        context_ls.append(single_context[0] + ":" + " ".join(single_context[1]) + "\n")
    context_str = " ".join(context_ls)
    full_input = context_str + "\nQuestion: " + input_data['question'] + "\nAnswer:" 
    return full_input

def inference(input_text, model):
    device = torch.device("cuda:0")
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])     
    outputs = model.generate(
            ids,
            #temperature=0.7,
            #do_sample = True,
            max_new_tokens = 15,
        )
    output_text = tokenizer.decode(outputs[0][length:])
    idx = output_text.find('.')
    output_text = output_text[:idx]
    return output_text

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hotpot_10k")
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--scale', type=str, default='3b')
    parser.add_argument('--result',type=str, default="Hotpot")
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,device_map='auto')

    data = []
    with open(f"../../dataset/HotpotQA/{args.dataset}.json",'r') as f:
        data = json.load(f)

    certain_data = []
    uncertain_data = []

    # sample[0] is question. sample[1] is answer.
    for sample in tqdm(data):
        output = inference(sample, model)
        if sample['answer'] in output:
            certain_data.append(sample)
        else:
            uncertain_data.append(sample)

    random.shuffle(certain_data)
    random.shuffle(uncertain_data)

    os.makedirs("../training_data",exist_ok=True)
    with open(f"../training_data/{args.result}_{model_name}_certain.json",'w') as f:
        json.dump(certain_data,f)
    with open(f"../training_data/{args.result}_{model_name}_uncertain.json",'w') as f:
        json.dump(uncertain_data,f)
