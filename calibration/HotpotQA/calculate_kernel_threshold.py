from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch.nn.functional as F
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

end_chars = ['.', '\n']
device = 'cuda:0'
llh_shift = torch.tensor(5.0)

def format_question(input_data):
    
    context_ls = []
    for single_context in input_data['context']:
        context_ls.append(single_context[0] + ":" + " ".join(single_context[1]) + "\n")
    context_str = " ".join(context_ls)
    full_input = context_str + "\nQuestion: " + input_data['question'] + "\nAnswer:" 
    return full_input

def von_neumann_entropy(kernel_matrix):
    # Compute eigenvalues of the kernel matrix
    eigenvalues = torch.linalg.eigvalsh(kernel_matrix)
    # Filter out eigenvalues close to 0 to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    # Compute von Neumann entropy
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
    return entropy
    
def calculate_certainty(input_data):

    weight_entailment = args.weight_entail  # Weight for the entailment class
    weight_neutral = args.weight_neutral  # Weight for the neutral class
    t = args.time  # Heat kernel diffusion time parameter

    answers = []
    occurance = {}
    full_input = format_question(input_data)
    question = input_data['question']
    inputs = tokenizer(full_input,return_tensors="pt").to(device)
    ids = inputs['input_ids']
    length = len(ids[0])
    generations = torch.ones((args.num_try, length + 15),
                                     dtype=torch.long,
                                     device=device)

    for i in range(args.num_try):
        generation = model.generate(
            ids,
            num_return_sequences=1,
            num_beams=1,
            temperature=0.7,
            do_sample = True,
            max_new_tokens = 15,
        )
        generations[i, :generation.shape[1]] = generation
        output_text = tokenizer.decode(generation[0][length:])
        idx = min([output_text.find(char) for char in end_chars if output_text.find(char) != -1] + [len(output_text)])
        output_text = output_text[:idx]
        answers.append(output_text)
    
    # Get unique generated texts
    unique_generated_texts = list(set(answers))
    
    # If only one unique answer is generated, directly return entropy as 0
    if len(unique_generated_texts) == 1:
        print("von neumann entropy: 0")
        return torch.tensor(0.0)
    
    # Initialize similarity matrix
    num_unique_texts = len(unique_generated_texts)
    similarity_matrix = torch.zeros((num_unique_texts, num_unique_texts), device=device)

    # Batch computation of similarities
    for i, qa_1 in enumerate(unique_generated_texts):
        for j in range(i + 1, num_unique_texts):
            qa_2 = unique_generated_texts[j]

            # Forward concatenation
            ori_input = qa_1 + ' [SEP] ' + qa_2
            encoded_input = deberta_tokenizer(ori_input, return_tensors="pt", padding=True).to(device)
            prediction = deberta_model(**encoded_input)['logits']
            predicted_probs = F.softmax(prediction, dim=1)
            forward_similarity = weight_entailment * predicted_probs[0, 2] + weight_neutral * predicted_probs[0, 1]

            # Reverse concatenation
            reverse_input = qa_2 + ' [SEP] ' + qa_1
            encoded_reverse_input = deberta_tokenizer(reverse_input, return_tensors="pt", padding=True).to(device)
            reverse_prediction = deberta_model(**encoded_reverse_input)['logits']
            reverse_predicted_probs = F.softmax(reverse_prediction, dim=1)
            reverse_similarity = weight_entailment * reverse_predicted_probs[0, 2] + weight_neutral * reverse_predicted_probs[0, 1]

            # Compute the average similarity
            similarity = (forward_similarity + reverse_similarity) / 2
            distance = 1 - similarity.item()

            # Update the similarity matrix
            similarity_matrix[i, j] = similarity_matrix[j, i] = distance    

    # Semantic kernel computation (Heat Kernel)
    kernel_matrix = torch.exp(-t * similarity_matrix)

    # Trace normalization and scaling
    diag_values = torch.sqrt(torch.diag(kernel_matrix).unsqueeze(0))  # Compute the diagonal elements
    kernel_matrix = kernel_matrix / (diag_values.T @ diag_values) / num_unique_texts

    # Compute von Neumann entropy
    entropy = von_neumann_entropy(kernel_matrix)
    print(f"von neumann entropy: {entropy.item()}")

    return -entropy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="uncertain")
    parser.add_argument('--model', type=str, default="openlm-research/open_llama_3b")
    parser.add_argument('--result',type=str, default="Hotpot")
    parser.add_argument("--num_try",type=int,default=10) #only required for uncertain method
    parser.add_argument("--alpha",type=float,default=0.8)
    parser.add_argument("--delta",type=float,default=0.01)
    parser.add_argument("--stored", action="store_true")
    parser.add_argument('--scale', type=str, default='3b')
    parser.add_argument('--weight_entail', type=float, default=0.7, help='Weight for the entailment class')
    parser.add_argument('--weight_neutral', type=float, default=0.3, help='Weight for the neutral class')
    parser.add_argument('--time', type=float, default=0.5, help='Heat kernel diffusion time parameter')
    
    args = parser.parse_args()
    
    model_name = args.model.split('/')[-1]
    if args.stored:
        with open(f"../training_data/Hotpot_{args.dataset}_{args.num_try}_{model_name}_kernel_certainties.json",'r') as f:
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
        tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,device_map=device,cache_dir=cache_dir)

        deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli",device_map=device)

        certainties = []
        with open(f"../training_data/Hotpot_{model_name}_{args.dataset}.json",'r') as f:
            data = json.load(f)
            
        # sample[0] is question. sample[1] is answer.
        for sample in tqdm(data):
            certainty = calculate_certainty(sample)
            certainties.append(certainty.item())

        with open(f"../training_data/Hotpot_{args.dataset}_{args.num_try}_{model_name}_kernel_certainties.json",'w') as f:
            json.dump(certainties,f)
    
    