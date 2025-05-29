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
import math
import os
import numpy as np

STOP = []
SURE = []
UNSURE = []


end_chars = ['.', '\n']
choices = ["A", "B", "C"]
candidate_answer = ['SUPPORTS.','REFUTES.','NOT ENOUGH INFO.']
mapping = {'SUPPORTS':"A",'REFUTES':"B",'NOT ENOUGH INFO':"C"}
llh_shift = torch.tensor(5.0)

def format_question(input_data):
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ': ' + candidate_answer[i]
    full_input += "\nAnswer:" 
    return full_input

def inference(input_text):
    full_input = format_question(input_text)+'(You should only output A,B or C)'
    # inputs = tokenizer(full_input,return_tensors="pt").to(device)
    # ids = inputs['input_ids']
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
    length = len(ids[0])     
    outputs = model.generate(
                **model_inputs,
                max_new_tokens = 1,
                output_scores = True,
                pad_token_id=tokenizer.eos_token_id,
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
    output_text = {0: "SUPPORTS.", 1: "REFUTES.", 2: "NOT ENOUGH INFO."}[np.argmax(probs)]
    conf = np.max(probs)
        
    return output_text, full_input, conf.item()

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

    for i in range(args.num_try):
        generation = model.generate(
            ids,
            num_return_sequences=1,
            num_beams=1,
            temperature=0.7,
            do_sample = True,
            max_new_tokens = 1,
        )
        generations[i, :generation.shape[1]] = generation
        output_text = tokenizer.decode(generation[0][length:])
        idx = min([output_text.find(char) for char in end_chars if output_text.find(char) != -1] + [len(output_text)])
        output_text = output_text[:idx]
        answers.append(output_text)
    
    average_neg_log_likelihoods = torch.zeros((generations.shape[0],)).cpu()
    semantic_set_ids = {}

    unique_generated_texts = list(set(answers))
    # print(unique_generated_texts)
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    
    list_of_semantic_set_ids = torch.tensor([semantic_set_ids[x] for x in answers])
    
    with torch.no_grad():
        for generation_index in range(generations.shape[0]):
            prompt = ids
            generation = generations[generation_index]
            target_ids = generation.clone()
            target_ids[:len(prompt)] = -100
            # print(generation.shape)
            model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
            average_neg_log_likelihood = model_output['loss'].cpu()
            average_neg_log_likelihoods[generation_index] = -average_neg_log_likelihood
    
    #print(average_neg_log_likelihoods)
    aggregated_likelihoods = []
    for semantic_set_id in torch.unique(list_of_semantic_set_ids):
        aggregated_likelihoods.append(torch.logsumexp(average_neg_log_likelihoods[list_of_semantic_set_ids == semantic_set_id], dim=0))
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
    entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
    #print(-entropy)
    return -entropy

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
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--result',type=str, default="fever")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--tau",type=float,default=0.5)
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-1]
    accelerator = Accelerator()
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',torch_dtype=torch.float16)
    model.bfloat16()
    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    THRESHOLD = args.tau

    with open(f"../../dataset/FEVER/fever_10k_test.json",'r') as f:
        data = json.load(f)

    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output, full_input, predict_conf = inference(sample)
            certainty = calculate_certainty(sample)
            #sure_prob = checksure(f"{full_input} {output}")
            result = (sample['label'] in output, predict_conf, certainty.item())
            print(sample['label'] in output, certainty.item())
            results.append(result)

    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.num_try}_semantic_{model_name}.json",'w') as f:
            json.dump(results,f)

                 
