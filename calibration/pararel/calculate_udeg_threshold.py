import numpy as np
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from scipy.stats import entropy
from scipy.special import comb, gammaln



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def inference(input_text, model, tokenizer):
    full_input = "Question:" + input_text + " Answer:"
    inputs = tokenizer(full_input, return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    outputs = model.generate(
            ids,
            temperature=0.7,
            do_sample=True,
            max_new_tokens=15,
        )
    
    output_text = tokenizer.decode(outputs[0][length:])
    idx = output_text.find('.')
    output_text = output_text[:idx] if idx != -1 else output_text
    return output_text

def jaccard_similarity(a: str, b: str) -> float:
    """
    Calculate Jaccard similarity between two text strings
    """
    # Split strings into word sets
    set_a = set(a.split())
    set_b = set(b.split())
    # Calculate intersection and union
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def compute_udeg(answers: list, similarity_func=jaccard_similarity) -> float:
    """
    Calculate UDeg uncertainty score based on a set of answers
    
    Parameters:
        answers: list of strings, each element is a generated answer
        similarity_func: function to calculate similarity between answers, default is jaccard_similarity
    
    Returns:
        U_Deg: calculated uncertainty score
    """
    m = len(answers)
    
    # Construct m x m similarity matrix W
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            W[i, j] = similarity_func(answers[i], answers[j])
    
    # Degree matrix D diagonal elements are the sum of each row
    degrees = np.sum(W, axis=1)
    
    # Calculate U_Deg according to formula: trace(m-D) = sum_{j=1}^{m}(m - D[j,j])
    U_Deg = (m * m - np.sum(degrees)) / (m * m)
    
    return U_Deg

def calculate_certainty(input_text, model, tokenizer, num_try):
    answers = []
    for i in range(num_try):
        output = inference(input_text, model, tokenizer)
        answers.append(output)
    
    # Calculate UDeg as the certainty measure (negative because higher UDeg means more uncertainty)
    udeg = compute_udeg(answers)
    return -udeg

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="uncertain")
    parser.add_argument('--model', type=str, default="openlm-research/open_llama_3b")
    parser.add_argument('--result', type=str, default="pararel")
    parser.add_argument("--num_try", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--stored", action="store_true")
    parser.add_argument('--scale', type=str, default='3b')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    model_name = args.model.split('/')[-1]
    
    if args.stored:
        with open(f"../training_data/pararel_{args.dataset}_{args.num_try}_{model_name}_udeg_certainties.json", 'r') as f:
            certainties = json.load(f)

        certainties.sort()
        n0 = len(certainties)
        total_sum = 0.0
        print(n0)
        for k in range(n0, 1, -1):
            log_comb = gammaln(n0 + 1) - (gammaln(k + 1) + gammaln(n0 - k + 1))
            log_term = log_comb + k * np.log(1 - args.alpha) + (n0 - k) * np.log(args.alpha)
            total_sum += np.exp(log_term)
            if total_sum > args.delta:
                print(k+1, certainties[k])
                if args.dataset == 'certain':
                    break
                with open(f"../training_data/{args.result}.txt", 'a') as f:
                    f.write(f"k:{k+1} threshold:{certainties[k]} alpha:{args.alpha} delta:{args.delta}\n")
                break
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='auto')

        certainties = []
        with open(f"../training_data/pararel_{model_name}_{args.dataset}.json", 'r') as f:
            data = json.load(f)

        # sample[0] is question. sample[1] is answer.
        for sample in tqdm(data):
            print(sample[0])
            certainty = calculate_certainty(sample[0], model, tokenizer, args.num_try)
            certainties.append(certainty)

        with open(f"../training_data/pararel_{args.dataset}_{args.num_try}_{model_name}_udeg_certainties.json", 'w') as f:
            json.dump(certainties, f)
