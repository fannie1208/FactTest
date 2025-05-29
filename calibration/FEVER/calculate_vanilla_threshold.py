from transformers import AutoTokenizer,AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
from scipy.special import comb, gammaln
import numpy as np
import math
import os



end_chars = ['.', '\n']
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

def inference(input_data):
    full_input = format_question(input_data)
    #full_input = input_text
    inputs = tokenizer(full_input,return_tensors="pt",padding=False).to(0)
    ids = inputs['input_ids']
    attention_mask = torch.ones_like(ids)
    #ids = inputs.input_ids
    length = len(ids[0])
    outputs = model.generate(
            ids,
            temperature=0.7,
            do_sample = True,
            max_new_tokens = 1,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
    
    output_text = tokenizer.decode(outputs[0][length:])
    idx = min([output_text.find(char) for char in end_chars if output_text.find(char) != -1] + [len(output_text)])
    output_text = output_text[:idx]
    return output_text

def calculate_certainty(input, model):
    answers = []
    occurance = {}
    uncertain_data = []
    for i in range(args.num_try):
        output = inference(input)
        output = output.strip('"')
        answers.append(output)
    
    for ans in answers:
        if ans in occurance:
            occurance[ans] += 1
        else:
            occurance[ans] = 1
    print(occurance)
    if occurance == {'':args.num_try}:
        freq_list = [1]*args.num_try
    else:
        freq_list = list(occurance.values())
    answer_entropy = entropy(freq_list)
    return -answer_entropy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="uncertain")
    parser.add_argument('--model', type=str, default="openlm-research/open_llama_3b")
    parser.add_argument('--result',type=str, default="FEVER")
    parser.add_argument("--num_try",type=int,default=5) 
    parser.add_argument("--alpha",type=float,default=0.8)
    parser.add_argument("--delta",type=float,default=0.01)
    parser.add_argument("--stored", action="store_true")
    parser.add_argument('--scale', type=str, default='3b')
    
    args = parser.parse_args()
    
    model_name = args.model.split('/')[-1]
    if args.stored:
        with open(f"../training_data/FEVER_{args.dataset}_{args.num_try}_{model_name}_certainties.json",'r') as f:
            certainties = json.load(f)

        certainties.sort()
        n0 = len(certainties)
        total_sum = 0.0
        print(n0)
        for k in range(n0, 1, -1):
            #log_comb = np.log(comb(n0, k))
            log_comb = gammaln(n0 + 1) - (gammaln(k + 1) + gammaln(n0 - k + 1))
            print(log_comb)
            log_term = log_comb + k * np.log(1 - args.alpha) + (n0 - k) * np.log(args.alpha)
            print(log_term)
            total_sum += np.exp(log_term)
            print(total_sum)
            if total_sum > args.delta:
                print(k+1, certainties[k])
                if args.dataset == 'certain':
                    break
                with open(f"../training_data/{args.result}.txt",'a') as f:
                    f.write(f"k:{k+1} threshold:{certainties[k]} alpha:{args.alpha} delta:{args.delta}\n")
                break

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='auto')
        model.bfloat16()
        certainties = []
        with open(f"../training_data/FEVER_{model_name}_{args.dataset}.json",'r') as f:
            data = json.load(f)

        # sample[0] is question. sample[1] is answer.
        for sample in tqdm(data):
            certainty = calculate_certainty(sample, model)
            print(certainty)
            certainties.append(certainty)

        with open(f"../training_data/FEVER_{args.dataset}_{args.num_try}_{model_name}_certainties.json",'w') as f:
            json.dump(certainties,f)
    
    