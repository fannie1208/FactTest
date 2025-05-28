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
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

STOP = []
SURE = []
UNSURE = []

cache_dir = '/work/vita/nie/cache/huggingface/hub'
end_chars = ['.', '\n']
choices = ["A", "B", "C"]
candidate_answer = ['supported.','partially_supported.','not_supported.']
mapping = {'supported':"A",'partially_supported':"B",'not_supported':"C"}
selfcheck_nli = SelfCheckNLI(device=torch.device("cuda"))

def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ':' + candidate_answer[i]
    full_input += "\nAnswer:" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])     
    outputs = model.generate(
                ids,
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
    output_text = {0: "supported", 1: "partially_supported", 2: "not_supported"}[np.argmax(probs)]
    conf = np.max(probs)
        
    return output_text, full_input, conf.item()

def certainty_inference(input_text):
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
            max_new_tokens = 1,
        )
    
    output_text = tokenizer.decode(outputs[0][length:])
    return output_text

def calculate_certainty(input_text):
    answers = []
    for i in range(args.num_try):
        output = certainty_inference(input_text)
        answers.append(output)
    
    sent_scores_nli = selfcheck_nli.predict(
        sentences = [answers[0]],                          # list of sentences
        sampled_passages = answers, # list of sampled passages
    )
    return sent_scores_nli[0]

def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
     #greedy decoding and calculate the confidence of sure and unsure
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--result',type=str, default="wice")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument("--tau",type=float,default=0.5)
    parser.add_argument('--scale', type=str, default='3b')
    
    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',torch_dtype=torch.float16, cache_dir=cache_dir)
    
    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    THRESHOLD = args.tau

    data = []

    with open(f"../../dataset/WiCE/wice_test.json",'r') as f:
        data = json.load(f)

    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output,full_input, predict_conf = inference(sample)
            certainty = calculate_certainty(sample)
            # sure_prob = checksure(f"{full_input} {mapping[output.strip('.')]}")
            result = (sample['label'] in output, predict_conf, certainty.item())
            print(sample['label'] in output, certainty.item())
            results.append(result)

    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.num_try}_scgpt_{model_name}.json",'w') as f:
            json.dump(results,f)

                 
