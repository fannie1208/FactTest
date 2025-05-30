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

end_chars = ['.', '\n']

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

def inference(input_text):
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])     
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

def certainty_inference(input_text, model):
    device = torch.device("cuda")
    full_input = format_question(input_text)
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
    idx = min([output_text.find(char) for char in end_chars if output_text.find(char) != -1] + [len(output_text)])
    output_text = output_text[:idx]
    return output_text

def calculate_certainty(input, model):
    answers = []
    occurance = {}
    uncertain_data = []
    for i in range(args.num_try):
        output = certainty_inference(input, model)
        output = output.strip('"')
        answers.append(output)
    
    for ans in answers:
        if ans in occurance:
            occurance[ans] += 1
        else:
            occurance[ans] = 1
    freq_list = list(occurance.values())
    answer_entropy = entropy(freq_list)
    return -answer_entropy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_7b')
    parser.add_argument('--result',type=str, default="Hotpot")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--tau",type=float,default=0.5)
    parser.add_argument('--seed', type=int, default=999, help='random seed')

    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    set_seed(args.seed)
    
    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',torch_dtype=torch.float16)
    
    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    THRESHOLD = args.tau

    data = {}
    prompt = {}
    with open(f"../../dataset/HotpotQA/hotpot_test.json",'r') as f:
        data = json.load(f)


    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output,full_input, predict_conf = inference(sample)
            certainty = calculate_certainty(sample, model)
            result = (sample['answer'].lower() in output.lower(), predict_conf, certainty.item())
            print(sample['answer'].lower() in output.lower(), certainty.item())
            results.append(result)

    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.num_try}_vanilla_{model_name}.json",'w') as f:
            json.dump(results,f)
        print('saved')

                 
