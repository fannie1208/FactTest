from transformers import AutoTokenizer,AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np

STOP = []
SURE = []
UNSURE = []


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def certainty_inference(input_text, model):
    device = torch.device("cuda:1")
    full_input = "Question:" + input_text + " Answer:"
    #full_input = input_text
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    #ids = inputs.input_ids
    length = len(ids[0])
    outputs = model.generate(
            ids,
            temperature=1,
            do_sample = True,
            max_new_tokens = 15,
        )
    
    output_text = tokenizer.decode(outputs[0][length:])
    idx = output_text.find('.')
    output_text = output_text[:idx]
    return output_text

def calculate_certainty(input_text, model):
    answers = []
    occurance = {}
    uncertain_data = []
    for i in range(args.num_try):
        output = certainty_inference(input_text, model)
        answers.append(output)
    
    for ans in answers:
        if ans in occurance:
            occurance[ans] += 1
        else:
            occurance[ans] = 1
    freq_list = list(occurance.values())
    answer_entropy = entropy(freq_list)
    return -answer_entropy

def inference(input_text):

    full_input = f"Question: {input_text} Answer:"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 15,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
    output_sequence = []
    product = 1
    count = 0
    for i in logits:        #greedy decoding and calculate the confidence
        pt = torch.softmax(torch.Tensor(i[0]),dim=0)
        max_loc = torch.argmax(pt)
        
        if max_loc in STOP:
            break
        else:
            output_sequence.append(max_loc)  
            product *= torch.max(pt)
            count += 1
            
    if output_sequence:
        output_text = tokenizer.decode(output_sequence)
    else:
        output_text = ""

    return output_text, full_input, np.power(product.item(),(1/count)).item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--result',type=str, default="pararel")
    parser.add_argument('--domain',type=str, default="ID")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument('--scale', type=str, default='3b')
    parser.add_argument('--seed', type=int, default=999, help='random seed')

    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    set_seed(args.seed)

    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')
    
    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)

    data = []
    with open(f"../../dataset/pararel/{args.domain}_test_pararel.json",'r') as f:
        data = json.load(f)
    
    # sample[0] is question. sample[1] is answer.
    with accelerator.split_between_processes(data) as data:
        results=[]
        certain_results = []
        uncertain_results = []

        for sample in tqdm(data):
            output, full_input, predict_conf = inference(sample[0])
            certainty = calculate_certainty(sample[0], model)
            result = (sample[1] in output,predict_conf,certainty)
            results.append(result)
        
    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.domain}_{args.num_try}_vanilla_{model_name}.json",'w') as f:
            json.dump(results,f)

                 
