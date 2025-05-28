from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np

cache_dir = '/work/vita/nie/cache/huggingface/hub'

choices = ["A", "B", "C"]
candidate_answer = ['SUPPORTS.','REFUTES.','NOT ENOUGH INFO.']
mapping = {'SUPPORTS':"A",'REFUTES':"B",'NOT ENOUGH INFO':"C"}

def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ': ' + candidate_answer[i]
    full_input += "\nAnswer:" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)
    #full_input = input_text
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])      
    outputs = model.generate(
            ids,
            #temperature=0.7,
            #do_sample = True,
            max_new_tokens = 1,
            output_scores = True,
            return_dict_in_generate=True
        )
    logits_for_choice = outputs['scores'][0][0]    #The first token
    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits_for_choice[tokenizer("A").input_ids[0]],        # 0 is bos_token
                    logits_for_choice[tokenizer("B").input_ids[0]],
                    logits_for_choice[tokenizer("C").input_ids[0]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    output_text = {0: "SUPPORTS.", 1: "REFUTES.", 2: "NOT ENOUGH INFO."}[np.argmax(probs)]
    return output_text, full_input

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="fever_10k")
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--result',type=str, default="FEVER")
    parser.add_argument('--scale', type=str, default='3b')
    
    args = parser.parse_args()
    
    model_name = args.model.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',cache_dir=cache_dir)

    data = []
    with open(f"../../dataset/FEVER/{args.dataset}.json",'r') as f:
        data = json.load(f)

    certain_data = []
    uncertain_data = []

    # sample[0] is question. sample[1] is answer.
    for sample in tqdm(data):
        output, full_input = inference(sample)
        gt = mapping[sample['label']]
        if sample['label'] in output:
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
