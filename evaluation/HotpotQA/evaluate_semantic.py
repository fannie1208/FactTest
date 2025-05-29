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
llh_shift = torch.tensor(5.0)
device=torch.device("cuda")

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
    inputs = tokenizer(full_input,return_tensors="pt").to(device)
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

def calculate_certainty(input_data):
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
    
    average_neg_log_likelihoods = torch.zeros((generations.shape[0],)).cpu()
    semantic_set_ids = {}

    unique_generated_texts = list(set(answers))
    # print(unique_generated_texts)
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    
    if len(unique_generated_texts) > 1:
        # Evalauate semantic similarity
        # print(len(unique_generated_texts))
        for i, reference_answer in enumerate(unique_generated_texts):
            for j in range(i + 1, len(unique_generated_texts)):

                qa_1 = unique_generated_texts[i]
                qa_2 = unique_generated_texts[j]

                ori_input = qa_1 + ' [SEP] ' + qa_2
                encoded_input = deberta_tokenizer.encode(ori_input, padding=True)
                prediction = deberta_model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = deberta_tokenizer.encode(reverse_input, padding=True)
                reverse_prediction = deberta_model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                if 2 in predicted_label or 2 in reverse_predicted_label:
                    semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
                else:
                    has_semantically_different_answers = True
                    deberta_prediction = 0

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
    
    print(average_neg_log_likelihoods)
    aggregated_likelihoods = []
    for semantic_set_id in torch.unique(list_of_semantic_set_ids):
        aggregated_likelihoods.append(torch.logsumexp(average_neg_log_likelihoods[list_of_semantic_set_ids == semantic_set_id], dim=0))
    aggregated_likelihoods = torch.tensor(aggregated_likelihoods) - llh_shift
    entropy = - torch.sum(aggregated_likelihoods, dim=0) / torch.tensor(aggregated_likelihoods.shape[0])
    print(-entropy)
    return -entropy


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_3b')
    parser.add_argument('--result',type=str, default="Hotpot")
    parser.add_argument("--num_try",type=int,default=5)
    parser.add_argument("--tau",type=float,default=0.5)
    parser.add_argument('--seed', type=int, default=999, help='random seed')

    args = parser.parse_args()
    model_name = args.model.split('/')[-1]
    set_seed(args.seed)
    accelerator = Accelerator()
    # device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',torch_dtype=torch.float16)
    model.bfloat16()
    deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
    deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli").cuda()

    STOP.append(tokenizer(".").input_ids)  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure").input_ids)
    UNSURE.append(tokenizer("unsure").input_ids)
    THRESHOLD = args.tau

    with open(f"../../dataset/HotpotQA/hotpot_test.json",'r') as f:
        data = json.load(f)


    with accelerator.split_between_processes(data) as data:
        results=[]

        for sample in tqdm(data):
            output,full_input, predict_conf = inference(sample)
            certainty = calculate_certainty(sample)
            #sure_prob = checksure(f"{full_input} {output}")
            result = (sample['answer'].lower() in output.lower(), predict_conf, certainty.item())
            print(sample['answer'].lower() in output.lower(), certainty.item())
            results.append(result)

    results=gather_object(results)
    
    if accelerator.is_main_process:
        os.makedirs("results",exist_ok=True)
        with open(f"results/ours_{args.num_try}_semantic_{model_name}.json",'w') as f:
            json.dump(results,f)

                 
