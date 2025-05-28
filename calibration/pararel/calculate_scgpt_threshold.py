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
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

cache_dir = '/work/vita/nie/cache/huggingface/hub'
#selfcheck_nli = SelfCheckNLI(device=torch.device("cuda"))

def inference(input_text, model):
    device = torch.device("cuda:0")
    full_input = "Question:" + input_text + " Answer:"
    #full_input = input_text
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    #ids = inputs.input_ids
    length = len(ids[0])
    outputs = model.generate(
            ids,
            temperature=0.7,
            do_sample = True,
            max_new_tokens = 15,
        )
    
    output_text = tokenizer.decode(outputs[0][length:])
    idx = output_text.find('.')
    output_text = output_text[:idx]
    return output_text

def calculate_certainty(input, model):
    answers = []
    for i in range(args.num_try):
        output = inference(input, model)
        answers.append(output)
    
    sent_scores_nli = selfcheck_nli.predict(
        sentences = [answers[0]],                          # list of sentences
        sampled_passages = answers, # list of sampled passages
    )
    return sent_scores_nli[0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="uncertain")
    parser.add_argument('--model', type=str, default="openlm-research/open_llama_3b")
    parser.add_argument('--result',type=str, default="pararel")
    parser.add_argument("--num_try",type=int,default=5) #only required for uncertain method
    parser.add_argument("--alpha",type=float,default=0.8)
    parser.add_argument("--delta",type=float,default=0.01)
    parser.add_argument("--stored", action="store_true")
    parser.add_argument('--scale', type=str, default='3b')
    
    args = parser.parse_args()
    
    model_name = args.model.split('/')[-1]
    # model_name = 'gpt_4o_mini'
    if args.stored:
        with open(f"../training_data/pararel_{args.dataset}_{args.num_try}_{model_name}_scgpt_certainties.json",'r') as f:
            certainties = json.load(f)
        certainties = [-i for i in certainties]
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
        tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='auto',cache_dir=cache_dir)

        certainties = []
        with open(f"../training_data/pararel_{model_name}_{args.dataset}.json",'r') as f:
            data = json.load(f)

        # sample[0] is question. sample[1] is answer.
        for sample in tqdm(data):
            print(sample[0])
            certainty = calculate_certainty(sample[0], model)
            certainties.append(certainty)

        with open(f"../training_data/pararel_{args.dataset}_{args.num_try}_{model_name}_scgpt_certainties.json",'w') as f:
            json.dump(certainties,f)
    
    