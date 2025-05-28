from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
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

cache_dir = '/work/vita/nie/cache/huggingface/hub'

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
choices = ["A", "B", "C"]
candidate_answer = ['supported.','partially_supported.','not_supported.']
mapping = {'supported':"A",'partially_supported':"B",'not_supported':"C"}
llh_shift = torch.tensor(5.0)
device = "cuda"

def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ':' + candidate_answer[i]
    full_input += "\nAnswer:" 
    return full_input

def calculate_certainty(input_data):
    answers = []
    occurance = {}
    full_input = format_question(input_data)
    inputs = tokenizer(full_input,return_tensors="pt").to(device)
    ids = inputs['input_ids']
    length = len(ids[0])
    generations = torch.ones((args.num_try, length + 1),
                                     dtype=torch.long,
                                     device=device)

    with torch.no_grad():
        for i in range(args.num_try):
            generation = model.generate(
                ids,
                num_return_sequences=1,
                num_beams=1,
                temperature=0.7,
                do_sample = True,
                max_new_tokens = 1,
            )
            generations[i, :generation.shape[1]] = generation[0]
            output_text = tokenizer.decode(generation[0][length:])
            answers.append(output_text)
    
    average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
    semantic_set_ids = {}

    unique_generated_texts = list(set(answers))
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    
    list_of_semantic_set_ids = torch.tensor([semantic_set_ids[x] for x in answers])

    with torch.no_grad():
        for generation_index in range(generations.shape[0]):
            prompt = ids
            generation = generations[generation_index]
            target_ids = generation.clone()
            target_ids[:len(prompt)] = -100
            print(generation.shape)
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
            average_neg_log_likelihood = model_output['loss']
            average_neg_log_likelihoods[generation_index] = -average_neg_log_likelihood
    
    aggregated_likelihoods = []
    for semantic_set_id in torch.unique(list_of_semantic_set_ids):
        aggregated_likelihoods.append(torch.logsumexp(average_neg_log_likelihoods[list_of_semantic_set_ids == semantic_set_id], dim=0))
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
    entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
    print(-entropy)
    return -entropy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="uncertain")
    parser.add_argument('--model', type=str, default="openlm-research/open_llama_3b")
    parser.add_argument('--result',type=str, default="WiCE")
    parser.add_argument("--num_try",type=int,default=10) #only required for uncertain method
    parser.add_argument("--alpha",type=float,default=0.8)
    parser.add_argument("--delta",type=float,default=0.01)
    parser.add_argument("--stored", action="store_true")
    
    args = parser.parse_args()
    
    model_name = args.model.split('/')[-1]
    # model_name = 'gpt_4o_mini'
    if args.stored:
        with open(f"../training_data/WiCE_{args.dataset}_{args.num_try}_{model_name}_semantic_certainties.json",'r') as f:
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
        deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli",device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='auto',cache_dir=cache_dir)

        certainties = []
        with open(f"../training_data/WiCE_{model_name}_{args.dataset}.json",'r') as f:
            data = json.load(f)
            
        # sample[0] is question. sample[1] is answer.
        for sample in tqdm(data):
            certainty = calculate_certainty(sample)
            print(certainty)
            certainties.append(certainty.item())

        with open(f"../training_data/WiCE_{args.dataset}_{args.num_try}_{model_name}_semantic_certainties.json",'w') as f:
            json.dump(certainties,f)