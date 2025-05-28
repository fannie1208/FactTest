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

cache_dir = '/work/vita/nie/cache/huggingface/hub'

end_chars = ['.', '\n']
choices = ["A", "B", "C"]
candidate_answer = ['SUPPORTS.','REFUTES.','NOT ENOUGH INFO.']
mapping = {'SUPPORTS':"A",'REFUTES':"B",'NOT ENOUGH INFO':"C"}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ': ' + candidate_answer[i]
    full_input += "\nAnswer:(You should only output A,B or C)" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": full_input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    ids = model_inputs.input_ids
    #inputs = tokenizer(full_input,return_tensors="pt",padding=False).to(device)
    #ids = inputs['input_ids']
    attention_mask = torch.ones_like(ids)
    length = len(ids[0])     
    outputs = model.generate(
                **model_inputs,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    # outputs = model.generate(
    #             ids,
    #             max_new_tokens = 1,
    #             output_scores = True,
    #             return_dict_in_generate=True,
    #             pad_token_id=tokenizer.eos_token_id,
    #             attention_mask=attention_mask
    #         )
    # print(outputs['scores'][0][0, 1065:1068])
    logits_for_choice = outputs['scores'][0][0]    #The first token
    #print(tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True))
    probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits_for_choice[tokenizer("A").input_ids[-1]],        # 0 is bos_token
                        logits_for_choice[tokenizer("B").input_ids[-1]],
                        logits_for_choice[tokenizer("C").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
    )
    output_text = {0: "SUPPORTS.", 1: "REFUTES.", 2: "NOT ENOUGH INFO."}[np.argmax(probs)]
    conf = np.max(probs)
    #print(np.argmax(probs), output_text)
    return output_text, full_input, conf.item()

def certainty_inference(input_text):
    full_input = format_question(input_text)
    #full_input = input_text
    inputs = tokenizer(full_input,return_tensors="pt",padding=False).to(device)
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
    return output_text

def calculate_certainty(input):
    answers = []
    occurance = {}
    uncertain_data = []
    for i in range(args.num_try):
        output = certainty_inference(input)
        answers.append(output)
    
    for ans in answers:
        if ans in occurance:
            occurance[ans] += 1
        else:
            occurance[ans] = 1
    if occurance == {'':args.num_try}:
        freq_list = [1]*args.num_try
    else:
        freq_list = list(occurance.values())
    answer_entropy = entropy(freq_list)
    return -answer_entropy

def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input,return_tensors="pt").to(device)
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
    print(UNSURE[0],unsure_prob)
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--result',type=str, default="FEVER")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--alpha",type=float,default=0.5)
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument("--tau",type=float,default=0.5)
    parser.add_argument('--scale', type=str, default='7b')
    parser.add_argument('--seed', type=int, default=999, help='random seed')

    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    #set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',cache_dir=cache_dir)
    model.bfloat16()
    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    THRESHOLD = args.tau

    data = {}
    prompt = {}
    with open(f"../../dataset/FEVER/fever_10k_test.json",'r') as f:
        data = json.load(f)


    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output, full_input, predict_conf = inference(sample)
            certainty = calculate_certainty(sample)
            # sure_prob = checksure(f"{full_input} {mapping[output[:-1]]}")
            result = (sample['label'] in output, predict_conf, certainty.item())
            print(sample['label'] in output, certainty.item())
            results.append(result)

    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.num_try}_vanilla_{model_name}.json",'w') as f:
            json.dump(results,f)
            print('\nsaved')

                 
