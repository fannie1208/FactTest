from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os

cache_dir = '/work/vita/nie/cache/huggingface/hub'

FALSE_RESPONSES = ["The answer is unknown.",
                   "The answer is uncertain.",
                   "The answer is unclear.",
                   "It is not known.",
                   "I do not know the answer.",
                   "I'm not sure.",
                   "There is no definitive answer.",
                   "There is much debate.",
                   "There is no concrete answer to this question.",
                   "It is impossible to answer.",
                   "There is no known case.",
                   "There is no public information available.",
                   "There is no scientific evidence.",
                   "There is no right answer.",
                   "It is impossible to know.",
                   "It is difficult to predict.",
                   ]

def inference(input_text, model):
    device = torch.device("cuda:0")
    full_input = "Question:" + input_text + " Answer:"
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
    parser.add_argument('--dataset', type=str, default="training_data")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="pararel")
    parser.add_argument('--scale', type=str, default='3b')
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,device_map='auto',cache_dir=cache_dir)

    data = []
    with open(f"../../dataset/pararel/{args.dataset}.json",'r') as f:
        data = json.load(f)

    certain_data = []
    uncertain_data = []

    # sample[0] is question. sample[1] is answer.
    for sample in tqdm(data):
        output = inference(sample[0], model)
        if sample[1] in output:
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
