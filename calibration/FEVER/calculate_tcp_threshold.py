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



torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
choices = ["A", "B", "C"]
candidate_answer = ['SUPPORTS.','REFUTES.','NOT ENOUGH INFO.']
mapping = {'SUPPORTS':"A",'REFUTES':"B",'NOT ENOUGH INFO':"C"}
llh_shift = torch.tensor(5.0)
device = "cuda"

def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ': ' + candidate_answer[i]
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
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--result',type=str, default="FEVER")
    parser.add_argument("--num_try",type=int,default=10) #only required for uncertain method
    parser.add_argument("--alpha",type=float,default=0.8)
    parser.add_argument("--delta",type=float,default=0.01)
    parser.add_argument("--stored", action="store_true")
    parser.add_argument('--scale', type=str, default='3b')
    
    args = parser.parse_args()
    
    model_name = args.model.split('/')[-1]
    if args.stored:
        with open(f"../training_data/FEVER_uncertain_{args.num_try}_{model_name}_semantic_certainties.json",'r') as f:
            uncertain_certainties = json.load(f)
        with open(f"../training_data/FEVER_certain_{args.num_try}_{model_name}_semantic_certainties.json",'r') as f:
            certain_certainties = json.load(f)
        uncertain_min = min(uncertain_certainties)
        print(min(certain_certainties), max(certain_certainties))
        certain_certainties = [uncertain_min - certainty for certainty in certain_certainties]

        all_scores = uncertain_certainties + certain_certainties
        n0 = len(all_scores)
        total_sum = 0.0
        print(uncertain_min, max(uncertain_certainties), n0)
        # Calculate quantile based on class-conditional conformal prediction
        # For scores where lower means more likely to be in set
        sorted_certainties = sorted(all_scores) # Sort in ascending order
        quantile_index = int(np.ceil((n0 + 1) * (1 - args.alpha))) - 1
        quantile_index = min(max(0, quantile_index), n0-1) # Ensure index is valid
        threshold = sorted_certainties[quantile_index]
        
        print(f"Quantile index: {quantile_index}, Threshold: {threshold}")
        # if args.dataset != 'certain':
        #     with open(f"../training_data/{args.result}.txt",'a') as f:
        #         f.write(f"quantile:{quantile_index} threshold:{threshold} alpha:{args.alpha}\n")
    
    else:
        deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli",device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='auto')
        model.bfloat16()

        certainties = []
        with open(f"../training_data/FEVER_{model_name}_{args.dataset}.json",'r') as f:
            data = json.load(f)
            
        # sample[0] is question. sample[1] is answer.
        for sample in tqdm(data):
            certainty = calculate_certainty(sample)
            print(certainty)
            certainties.append(certainty.item())

        with open(f"../training_data/FEVER_{args.dataset}_{args.num_try}_{model_name}_semantic_certainties.json",'w') as f:
            json.dump(certainties,f)
    
    